import sys
import numpy as np
import pandas as pd
import torch
import os
import math

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from calldb import CallDB
from polarity import Polarity
import DDI_runner
import BA_runner




def main():
    req_id = sys.argv[1]
    run = Run(req_id)
    run.run()


class Run:
    def __init__(self, req_id):
        self.req_id = req_id
        
        self.suitability = CallDB("pro_carrier_suitability_assessment")
        self.interaction = CallDB("pro_drug_drug_interaction")
        
        self.request = self.suitability.from_db(f"""
        SELECT * 
        FROM pro_carrier_suitability_assessment.request 
        WHERE req_id = "{self.req_id}"
        ;
        """)
        
        # interaction_dict table fetch
        self.inter_dict = self.interaction.from_db("""
        SELECT *
        FROM pro_drug_drug_interaction.interaction_dictionary
        ;
        """)
        
        self.drug_name = self.request.loc[0, 'drug_name']
        self.smiles = self.request.loc[0, 'smiles']
        self.protein_name = self.request.loc[0, 'protein_name']
        self.sequence = self.request.loc[0, 'sequence']
        self.weight = self.request.loc[0, 'weight']
        
        self.weight_split = self.weight.split('|')
        self.ba_wt = float(self.weight_split[0])
        self.ae_wt = float(self.weight_split[1])
        self.ddi_wt = float(self.weight_split[2])
        
        
    def run(self):     
        drug_fp = self.drug_encoding(self.smiles)
        target_embd = self.target_encoding(self.sequence)
        
        polarity = self.pol(self.smiles)
        self.pol_candidate(polarity)    # 예측극성에 해당되는 후보 fetch -> self
        carrier_fp = self.pol_carrier_fp()
        
        ddi_pred = self.ddi(drug_fp, carrier_fp)
        self.ddi_candidate(ddi_pred)
        carrier_fp = self.ddi_carrier_fp()
        
        ba_pred = self.ba(carrier_fp, target_embd)
        
        result = self.make_df()
        
        self.to_result(result)
        self.to_result_ref(result)
        self.to_img_dir()       
        
        
    def drug_encoding(self, smiles: str) -> torch.Tensor:
        drug_m = Chem.MolFromSmiles(smiles)
        drug_fp = torch.tensor(np.array(AllChem.GetMorganFingerprintAsBitVect(drug_m, 3, nBits=1024)), dtype=torch.float32)
        return drug_fp
    
    
    def target_encoding(self, sequence:str) -> torch.Tensor:
        amino_dict = {s: i for i, s in enumerate(list('ACDEFGHIKLMNOPQRSTVWXY'))}
        sequence_embd = [amino_dict[c] for c in sequence.upper()]  # 단백질 시퀀스 숫자로 변환

        max_len = 10000
        for i in range(max_len - len(sequence_embd)):
            sequence_embd.append(0)

        target_embd = torch.tensor(sequence_embd, dtype=torch.float32)[:2048].view(1, 2048)
        return target_embd   # 벡터
    
    
    def pol(self, smiles):
        polarity = Polarity(smiles)    # polarity 예측
        return polarity

    def pol_candidate(self, polarity):
        # 극성에 대응하는 carier_candidate fetch
        pol_matched = self.suitability.from_db(f"""
        SELECT Drugbank_ID, name, finger_print, block_type, block_score, SMILES
        FROM pro_carrier_suitability_assessment.block_type_library 
        WHERE block_type = {polarity};
        """)

        self.pol_matched = pol_matched[['Drugbank_ID',
                                   'name',
                                   'finger_print',
                                   'block_type',
                                   'block_score',
                                   'SMILES']]
        
    
    def pol_carrier_fp(self) -> torch.Tensor:
        # fingerprint 전처리
        carrier_fp = []
        for i, (_, _, fp, _, _, _) in self.pol_matched.iterrows():
            carrier_fp.append(np.fromstring(fp.replace('', ' '), dtype=int, sep=' '))

        carrier_fp = torch.tensor(carrier_fp, dtype=torch.float32)
        return carrier_fp
    
    
    def ddi(self, drug_fp: torch.Tensor, carrier_fp: torch.Tensor):
        ddi_pred = DDI_runner.run(drug_fp, carrier_fp)
        ddi_pred = list(np.array(ddi_pred.detach()))
        return ddi_pred

        
    def ddi_candidate(self, ddi_pred) -> object:
        # interaction 점수 가져오기
        self.ddi_type = pd.DataFrame(np.array(self.inter_dict[['label', 'interaction_type']].iloc[ddi_pred]), columns=['label', 'interaction_type']) 

        
        if self.ddi_wt == 1:
            inter_type = 2

        elif self.ddi_wt == 0.5:
            inter_type = 1

        else:
            inter_type = 0
            
        
        # 만족하는 점수만 선택
        self.ddi_matched_type = self.ddi_type['interaction_type'] >= inter_type
        self.ddi_matched = self.pol_matched[['Drugbank_ID', 'finger_print', 'SMILES']].iloc[list(self.ddi_type.index[self.ddi_matched_type])]

        
    # string to np.ndarray
    def ddi_carrier_fp(self) -> torch.Tensor:  
        carrier_fp = []
        for i, (id, fp, _) in self.ddi_matched.iterrows():
            carrier_fp.append(np.fromstring(fp.replace('', ' '), dtype=int, sep=' '))

        carrier_fp = torch.tensor(carrier_fp, dtype=torch.float32)
        return carrier_fp
    
    # Binding_affinity 예측
    def ba(self, carrier_fp: torch.Tensor, target_embd: torch.Tensor) -> object:

        # 같은 개수만큼 복사
        sequence_temp = torch.tensor([], dtype=torch.float32).view(0, 2048)
        for i in range(len(carrier_fp)):
            sequence_temp = torch.cat([sequence_temp, target_embd], dim=0)
        target_embd = sequence_temp

        ba_pred = BA_runner.run(carrier_fp, target_embd, len(carrier_fp))
        ba_pred = np.round(np.array(ba_pred.detach()), 4)
        self.ba_pred = np.where(ba_pred < 0, 0.0001, ba_pred)
        
        return self.ba_pred

    
    def make_df(self) -> object:
        drugbank_id = pd.DataFrame(self.ddi_matched['Drugbank_ID'], columns=['Drugbank_ID'])
        result = drugbank_id
        pol_result = self.pol_matched

        ddi_result1 = self.ddi_type[self.ddi_matched_type]

        ddi_result2 = self.inter_dict.iloc[list(self.ddi_type[self.ddi_matched_type]['label'])][['label', 'interaction']]
        ddi_result2 = pd.DataFrame(np.array(ddi_result2), columns=['label', 'interaction'])

        ba_result = pd.DataFrame(self.ba_pred, columns=['ic50', 'ec50'])

        request_id = pd.DataFrame({'req_id' : [self.req_id for i in range(len(result))]})

        result = result.join(ddi_result1).merge(pol_result).join(ba_result).join(ddi_result2[['interaction']]).join(request_id)
        result = result[['req_id', 'Drugbank_ID', 'name', 'block_type', 'block_score', 'interaction', 'interaction_type', 'ic50', 'ec50']]

        return result
    
    
    def to_result(self, result: object):
        for idx, (req_id, Drugbank_ID, name, block_type, block_score, interaction, interaction_type, ic50, ec50) in result.iterrows():
            name = name.replace("'", "`")
            
            interaction = interaction.replace('A', self.drug_name).replace('B', name)

            n_ref_ae = self.suitability.from_db(f"""
            SELECT COUNT(*)
            FROM pro_carrier_suitability_assessment.abstract_adverse_effect
            WHERE abstract like '%{name}%'
            """
                                          ).iloc[0,0]

            total_score = (20 * block_score) + (10 * interaction_type) + (-math.log(ic50)) + (-math.log(ec50))

            self.suitability.query_db(f"""
            INSERT INTO pro_carrier_suitability_assessment.result(req_id, 
                                                                  idx, 
                                                                  Drugbank_ID, 
                                                                  name, 
                                                                  block_type, 
                                                                  block_score, 
                                                                  interaction, 
                                                                  interaction_type, 
                                                                  ic50, ec50, 
                                                                  n_ref_da, 
                                                                  n_ref_ae,
                                                                  total_score)
            VALUES ('{req_id}', 
                     {idx}, 
                     '{Drugbank_ID}', 
                     '{name}', 
                     {block_type}, 
                     {round(block_score, 3)}, 
                     '{interaction}', 
                     {interaction_type}, 
                     {round(ic50, 3)}, 
                     {round(ec50, 3)},
                     0, 
                     {n_ref_ae},
                     {total_score});    
            """)
    

    def to_result_ref(self, result: object):
        for idx, (req_id, Drugbank_ID, name) in result[['req_id', 'Drugbank_ID', 'name']].iterrows():
            name = name.replace("'", "`")
            self.suitability.query_db(f"""
            INSERT INTO pro_carrier_suitability_assessment.result_adverse_effects_ref (req_id, index_id, reference_title, year, summary)
            (
            SELECT '{req_id}', {idx}, title, year, summary
            FROM pro_carrier_suitability_assessment.abstract_adverse_effect
            WHERE abstract like '%{name}%'
            );
            """)
    
    
    # 분자 이미지 저장
    def to_img_dir(self):
        smiles = pd.DataFrame(self.ddi_matched[['Drugbank_ID', 'SMILES']])
        os.system(f'mkdir /BiO/projects/polarity/carrier_suitability_assessment/img_result/{self.req_id}')
        for i, (id, smi) in smiles.iterrows():
            m = Chem.MolFromSmiles(smi)
            Draw.MolToFile(m, f'/BiO/projects/polarity/carrier_suitability_assessment/img_result/{self.req_id}/{id}.png')
        
        
if __name__ == "__main__":
    main()
