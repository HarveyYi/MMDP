import os.path as osp
import pandas as pd
import numpy as np


from typing import Any, Callable, Optional, Tuple

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


import torch
from mmdp.utils import listdir_nohidden, verify_str_arg

from .build import DATASET_REGISTRY
from .base import Datum, DatasetBase

OMIC_MODALITIES = ['rna']  


def _series_intersection(s1, s2):
    r"""
    Return insersection of two sets
    
    Args:
        - s1 : set
        - s2 : set 
    
    Returns:
        - pd.Series
    
    """
    return pd.Series(list(set(s1) & set(s2)))



@DATASET_REGISTRY.register()
class Survival(DatasetBase):
    """Survival
    """
    def __init__(self, cfg, fold=None):
        self.cfg = cfg
        if fold == None:
            fold = self.cfg.DATASET.FOLD
        else:
            fold = fold
        self.data_name = self.cfg.DATASET.NAME
        self.urvival_endpoint = self.cfg.DATASET.SURVIVAL_ENDPOINT
        
        self.omics_type = self.cfg.DATASET.OMIC.TYPE
        self.pathwayname = self.cfg.DATASET.OMIC.PATHWAY
        self.wsi_feature = self.cfg.DATASET.PATH.FEATURE
        
        self.root = osp.abspath(osp.expanduser(self.cfg.DATASET.ROOT))
        
        
        self._meta_folder = osp.join(self.root, "metadata") 
        self._meta_path = osp.join(self._meta_folder, f"tcga_{self.data_name}.csv") 

        self._histo_folder = osp.join(self.root, f"wsi_data/{self.data_name}/{self.wsi_feature}")  
        self._pathomic_folder = osp.join(self.root, f"pathway_features/{self.data_name}")  
        self._omic_folder = osp.join(self.root, f"raw_rna_data/{self.pathwayname}/{self.data_name}") 
        
        # 5 fold 
        # import pdb;pdb.set_trace()
        self._fold = verify_str_arg(str(fold), "Fold", ("0", "1", "2", "3", "4"))
        self._split_fold = osp.join(self.root, f"5foldcv/tcga_{self.data_name}/splits_{self._fold}.csv") 
 
        # import pdb;pdb.set_trace()
        self.classnames_dict = {0:"S I", 1:"S II", 2:"S III", 3:"S IV"}
        
        
        if self.urvival_endpoint == "OS":
            event_time_var = "survival_months"
            censorship_var = "censorship"
        elif self.urvival_endpoint == "PFI":
            event_time_var = "survival_months_pfi"
            censorship_var = "censorship_pfi" 
        elif self.urvival_endpoint == "DSS":
            event_time_var = "survival_months_dss"
            censorship_var = "censorship_dss"
        
        self._read_omics_data()
        if self.omics_type == "group":
            self._setup_omics_group()
        elif self.omics_type == "pathway":
            self._setup_omics_pathway()
        else:
            # "all" type
            self.omic_names = []
            self.omic_sizes = []
            self.omic_groups = 1
        

        
        self.meta_data = pd.read_csv(self._meta_path, low_memory=False)

        uncensored_df = self._clean_label_data(censorship_var)
        self._discretize_survival_months(uncensored_df, event_time_var)
        
        train, scaler = self._read_data(
                event_time_var=event_time_var, censorship_var=censorship_var, split= "train", scaler=None
        )
        
        val, scaler  = self._read_data(
                event_time_var=event_time_var, censorship_var=censorship_var, split= "val", scaler=scaler
        )
        
        test = val



        if len(val) == 0:
            val = None

        super().__init__(train=train,  val=val, test=test)

    def _clean_label_data(self, censorship_var):
        r"""
        Clean the metadata. For breast, only consider the IDC subtype.
        
        Args:
            - self
            - censorship_var
        
        Returns:
            - None
            
        """

        if "IDC" in self.meta_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            self.meta_data = self.meta_data[self.meta_data['oncotree_code'] == 'IDC']

        self.patients_df = self.meta_data.drop_duplicates(['case_id']).copy()
        uncensored_df = self.patients_df[self.patients_df[censorship_var] < 1]
        
        return uncensored_df
    
    def _discretize_survival_months(self, 
                                    uncensored_df, 
                                    event_time_var,
                                    n_bins=4, 
                                    eps=1e-6):
        r"""
        This is where we convert the regression survival problem into a classification problem. We bin all survival times into 
        quartiles and assign labels to patient based on these bins.
        
        Args:
            - self
            - uncensored_df : pd.DataFrame
            - event_time_var : str
            - n_bins : int
            - eps : Float 
            
        
        Returns:
            - None 
        
        """
        # cut the data into self.n_bins (4= quantiles)
        disc_labels, q_bins = pd.qcut(uncensored_df[event_time_var], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = self.meta_data[event_time_var].max() + eps
        q_bins[0] = self.meta_data[event_time_var].min() - eps
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.patients_df[event_time_var], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patients_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins


    def _read_omics_data(self):
        r"""
        read the csv with the omics data
        
        Args:
            - self
        
        Returns:
            - None
        
        """
        self.omic_modalities = {}
        for modality in OMIC_MODALITIES:
            self.omic_modalities[modality] = pd.read_csv(
                osp.join(self._omic_folder, f"{modality}_clean.csv"),
                engine='python',
                index_col=0
            )    
    
    def _setup_omics_group(self):
        r"""
        Process the signatures for the 6 functional groups
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        self.signatures = pd.read_csv(
            osp.join(self._meta_folder, "signatures.csv"))
        
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.omic_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]   
        self.omic_groups = len(self.omic_names)   
    

    def _setup_omics_pathway(self):

        r"""
        Process the signatures for the 331 pathways 
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        # running with hallmarks, reactome, or combined signatures
        self.signatures = pd.read_csv(
            osp.join(self._meta_folder, f"{self.pathwayname}_signatures.csv"))
        
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.omic_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]   
        self.omic_groups = len(self.omic_names)   

    def _get_scaler(self, data):
        r"""
        Define the scaler for training dataset. Use the same scaler for validation set
        
        Args:
            - self 
            - data : np.array

        Returns: 
            - scaler : MinMaxScaler
        
        """
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        return scaler
    
    def _apply_scaler(self, data, scaler):
        r"""
        Given the datatype and a predefined scaler, apply it to the data 
        
        Args:
            - self
            - data : np.array 
            - scaler : MinMaxScaler 
        
        Returns:
            - data : np.array """
        
        # find out which values are missing
        zero_mask = data == 0

        # transform data
        transformed = scaler.transform(data)
        data = transformed

        # rna -> put back in the zeros 
        data[zero_mask] = 0.
        
        return data

    def _read_data(self, 
                   event_time_var = "survival_months_dss",
                   censorship_var = "censorship_dss",
                   split = "train", 
                   key = "rna",
                   scaler=None,
                   ):
        
        items = []
        if not scaler:
            scaler = {}
            
            
        splits = pd.read_csv(self._split_fold)
        splits_ids = splits[split].dropna().reset_index(drop=True)
        
        
        omics_raw_data = self.omic_modalities[key]
        mask = omics_raw_data.index.isin(splits_ids.tolist())

        # normlize train omics data
        filtered_omics_raw_data = omics_raw_data[mask]
        filtered_omics_raw_data = filtered_omics_raw_data[~filtered_omics_raw_data.index.duplicated()] # drop duplicate case_ids
        filtered_omics_raw_data["temp_index"] = filtered_omics_raw_data.index
        filtered_omics_raw_data.reset_index(inplace=True, drop=True)
        
        case_ids = filtered_omics_raw_data["temp_index"]
        mask2 = [True if item in list(case_ids) else False for item in splits_ids]
        splits_ids = splits_ids[mask2]
        splits_ids.reset_index(inplace=True, drop=True)
        # import pdb;pdb.set_trace()
        
        filtered_omics_normed_data = filtered_omics_raw_data
        data_for_norm = filtered_omics_raw_data.drop(labels="temp_index", axis=1)
        # store original num_patients and num_feats 
        num_patients = data_for_norm.shape[0]
        num_feats = data_for_norm.shape[1]
        columns = {}
        for i in range(num_feats):
             columns[i] = data_for_norm.columns[i]

        if split == "val":
            # flatten the df into 1D array (make it a column vector)
            flat_data = np.expand_dims(data_for_norm.values.flatten(), 1)
            
            # get scaler
            scaler_for_data = scaler[key]

            # normalize 
            normed_flat_data = self._apply_scaler(data = flat_data, scaler = scaler_for_data)

            # change 1D to 2D
            filtered_omics_normed_data = pd.DataFrame(normed_flat_data.reshape([num_patients, num_feats]))

            # add in case_ids
            filtered_omics_normed_data["temp_index"] = case_ids
            filtered_omics_normed_data.rename(columns=columns, inplace=True)       
            
        elif  split == "train":
            # flatten the df into 1D array (make it a column vector)
            flat_data = data_for_norm.values.flatten().reshape(-1, 1)
            
            # get scaler
            scaler_for_data = self._get_scaler(flat_data)

            # normalize 
            normed_flat_data = self._apply_scaler(data = flat_data, scaler = scaler_for_data)

            # change 1D to 2D
            filtered_omics_normed_data = pd.DataFrame(normed_flat_data.reshape([num_patients, num_feats]))

            # add in case_ids
            filtered_omics_normed_data["temp_index"] = case_ids
            filtered_omics_normed_data.rename(columns=columns, inplace=True)

            # store scaler
            scaler[key] = scaler_for_data
        
        for patient_id in splits_ids:
            rows = self.meta_data.loc[self.meta_data['case_id'] == patient_id]
            label = self.patients_df.loc[self.meta_data['case_id'] == patient_id, 'label'].values[0]
            event_time = self.patients_df.loc[self.meta_data['case_id'] == patient_id, event_time_var].values[0]
            censorship = self.patients_df.loc[self.meta_data['case_id'] == patient_id, censorship_var].values[0]
            
            single_omics_data = filtered_omics_normed_data[filtered_omics_normed_data["temp_index"]== patient_id]
            single_omics_data = single_omics_data.drop(columns="temp_index")
            
            
            if self.omics_type == "group" or self.omics_type == "pathway":
                omics_data = []
                for i in range(self.omic_groups):
                    omics_data.append(torch.tensor(single_omics_data[self.omic_names[i]].values[0]).float())    
            else:
                single_omics_data = single_omics_data.reindex(sorted(single_omics_data.columns), axis=1)
                omics_data = [torch.squeeze(torch.Tensor(single_omics_data.values)).float()]            
            
            
            histo_paths = []
            for i, row in rows.iterrows():
                slide_id = row["slide_id"]
                slide_name = slide_id.replace('svs', 'pt')
                histo_paths.append(osp.join(self._histo_folder, f"{slide_name}"))
                
            surv_label = {"label": int(label), "event_time": event_time, "censorship": censorship}

                    
            item = Datum(
                        patient_id=patient_id,
                        histo_paths=histo_paths, 
                        omics_data=omics_data,
                        classname=self.classnames_dict[label],
                        label=surv_label, 
                        )
            items.append(item)
                
        return  items, scaler


    
    
