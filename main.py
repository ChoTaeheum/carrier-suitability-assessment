import sys
import os

from calldb import CallDB
from polarity import Polarity
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import DDI_runner
import BA_runner


def main():
    
    req_id = sys.argv[1]
    
    # req_id를 통해 사용자 입력 파싱
    suitability = CallDB("pro_carrier_suitability_assessment")
    request = suitability.from_db(f"""
    SELECT * 
    FROM pro_carrier_suitability_assessment.request 
    WHERE req_id = "{req_id}"
    ;
    """)

    
    drug_name = request.loc[0, 'drug_name']
    smiles = request.loc[0, 'smiles']
    protein_name = request.loc[0, 'protein_name']
    sequence = request.loc[0, 'sequence']
    weight = request.loc[0, 'weight']

    ##################
    # 타겟 약물이 친수성인지 소수성인지 확인
    polarity = Polarity(smiles)

    ##################
    # 타겟 약물 인코딩
    try:
        drug_m = Chem.MolFromSmiles(smiles)
        drug_fp = torch.tensor(np.array(AllChem.GetMorganFingerprintAsBitVect(drug_m, 3, nBits=1024)), dtype=torch.float32)
    
    except:
        report_error_and_exit('약물 인코딩 실패')

    ##################
    # 전달체 후보군 finger print 불러오기
    call_db = CallDB('pro_carrier_suitability_assessment')
    pol_matched = call_db.from_db(f"""
    SELECT Drugbank_ID, name, finger_print, block_type, block_score, SMILES
    FROM pro_carrier_suitability_assessment.block_type_library 
    WHERE block_type = {polarity}
    ;
    """)
    pol_matched = pol_matched[['Drugbank_ID', 'name', 'finger_print', 'block_type', 'block_score', 'SMILES']]

    carrier_fp = []
    for i, (_, _, fp, _, _, _) in pol_matched.iterrows():
        carrier_fp.append(np.fromstring(fp.replace('', ' '), dtype=int, sep=' '))

    carrier_fp = torch.tensor(carrier_fp, dtype=torch.float32)

    ##################
    # DDI 분석 실행
    ddi_pred = DDI_runner.run(drug_fp, carrier_fp)
    ddi_pred = list(np.array(ddi_pred.detach()))


    ####################
    # DDI 결과를 바탕으로 BA 계산할 전달체 후보 추출
    interaction = CallDB("pro_drug_drug_interaction")
    inter_dict = interaction.from_db("""
    SELECT *
    FROM pro_drug_drug_interaction.interaction_dictionary
    ;
    """)


    ddi_type = pd.DataFrame(np.array(inter_dict[['label', 'interaction_type']].iloc[ddi_pred]), columns=['label', 'interaction_type'])   # interaction 점수 가져오기
    ddi_matched = pol_matched[['Drugbank_ID', 'finger_print', 'SMILES']].iloc[list(ddi_type.index[ddi_type['interaction_type']==2])]   # 만족하는 점수만 선택

    carrier_fp = []    # carrier_fp 여기서 다시 초기화
    for i, (id, fp, _) in ddi_matched.iterrows():
        carrier_fp.append(np.fromstring(fp.replace('', ' '), dtype=int, sep=' '))

    carrier_fp = torch.tensor(carrier_fp, dtype=torch.float32)



    #######################
    # 타겟 단백질 인코딩
    amino_dict = {s: i for i, s in enumerate(list('ABCDEFGHIKLMNOPQRSTUVWXYZ'))}
    
    try:
        sequence_embd = [amino_dict[c] for c in sequence.upper()]

        max_len = 10000
        for i in range(max_len - len(sequence_embd)):
            sequence_embd.append(0)

        sequence_embd = torch.tensor(sequence_embd, dtype=torch.float32)[:2048].view(1, 2048)
        sequence_temp = torch.tensor([], dtype=torch.float32).view(0, 2048)

        for i in range(len(carrier_fp)):
            sequence_temp = torch.cat([sequence_temp, sequence_embd], dim=0)
        sequence_embd = sequence_temp
        
    except:
        report_error_and_exit('타겟 인코딩 실패')

    #######################
    # Binding_affinity 예측

    ba_pred = BA_runner.run(carrier_fp, sequence_embd, len(carrier_fp))
    ba_pred = np.round(np.array(ba_pred.detach()), 4)
    ba_pred = np.where(ba_pred < 0, 0, ba_pred)


    ######################
    # DB_result에 삽입할 DF 만들기
    drugbank_id = pd.DataFrame(ddi_matched['Drugbank_ID'], columns=['Drugbank_ID'])

    result = drugbank_id
    pol_result = pol_matched

    ddi_result1 = ddi_type[ddi_type['interaction_type']==2]

    ddi_result2 = inter_dict.iloc[list(ddi_type[ddi_type['interaction_type']==2]['label'])][['label', 'interaction']]
    ddi_result2 = pd.DataFrame(np.array(ddi_result2), columns=['label', 'interaction'])

    ba_result = pd.DataFrame(ba_pred, columns=['ic50', 'ec50'])

    request_id = pd.DataFrame({'req_id' : [req_id for i in range(len(result))]})

    # 결과 취합
    result = result.join(ddi_result1).merge(pol_result).join(ba_result).join(ddi_result2[['interaction']]).join(request_id)
    result = result[['req_id', 'Drugbank_ID', 'name', 'block_type', 'block_score', 'interaction', 'interaction_type', 'ic50', 'ec50']]

    #######################
    # 반복문 돌면서 insert 문 실행
    for idx, (req_id, Drugbank_ID, name, block_type, block_score, interaction, interaction_type, ic50, ec50) in result.iterrows():
        interaction = interaction.replace('A', drug_name).replace('B', name)
        
        n_ref_ae = suitability.from_db(f"""
        SELECT COUNT(*)
        FROM pro_carrier_suitability_assessment.abstract_adverse_effect
        WHERE abstract like '%{name}%'
        """
                                  ).iloc[0,0]
        
        total_score = (20 * block_score) + (10 * interaction_type) + (-math.log(ic50)) + (-math.log(ec50))
            
        suitability.query_db(f"""
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
        
    # 부작용 논문 검색후 DB 저장        
    for idx, (req_id, Drugbank_ID, name) in result[['req_id', 'Drugbank_ID', 'name']].iterrows():
        suitability.query_db(f"""
        INSERT INTO pro_carrier_suitability_assessment.result_adverse_effects_ref (req_id, index_id, reference_title, year, summary)
        (
        SELECT '{req_id}', {idx}, title, year, summary
        FROM pro_carrier_suitability_assessment.abstract_adverse_effect
        WHERE abstract like '%{name}%'
        );
        """)

    # 분자 이미지 저장
    smiles = pd.DataFrame(ddi_matched[['Drugbank_ID', 'SMILES']])
    os.system(f'mkdir /BiO/projects/polarity/carrier_suitability_assessment/img_result/{req_id}')
    for i, (id, smi) in smiles.iterrows():
        m = Chem.MolFromSmiles(smi)
        Draw.MolToFile(m, f'/BiO/projects/polarity/carrier_suitability_assessment/img_result/{req_id}/{id}.png')
        
        
if __name__ == "__main__":
    main()






























