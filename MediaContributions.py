import numpy as np 
import pandas as pd

import json

from LinearRegression import LR
from HillsTransformation import HillsTransformation


import matplotlib.pyplot as plt


class MediaContributionCalculator: 
    obligatory_sections = ['Target variables', 'Campaign variables']
    campaign_var_definition = ['Name', 'Alpha', 'Gamma']
    version = 1.001
    
    def __init__(self, p_value=0.05, correct_base=False): 
        self.p_value = p_value
        self.correct_base = correct_base
    
    def CheckFixSpec(self, model_data, model_spec): 
        # проверка обязательных секций
        for itm in self.obligatory_sections: 
            assert itm in model_spec.keys(),  "Ошибка спецификации модели: нет " + itm
            
        # проверка Target variables
        # обязательная секция
        # может быть строкой - если один таргерт
        # может быть непустым списком
        if isinstance(model_spec['Target variables'], str):
            model_spec['Target variables'] = [model_spec['Target variables']]
        elif isinstance(model_spec['Target variables'], list): 
            assert len(model_spec['Target variables']) > 0, "Ошибка спецификации модели: Target variables пустой"
        else: 
            assert False, "Ошибка спецификации модели: Target variables неверный формат"
            
        for x in model_spec['Target variables']: 
            assert x in model_data.columns, "Ошибка спецификации модели: Target variable '" + x + "' нет в данных"

        # проверка Campaign variables
        # обязательная секция
        # может быть списком, может быть пустым, или None
        if model_spec['Campaign variables'] is None: 
            model_spec['Campaign variables'] = []
        elif isinstance(model_spec['Target variables'], list): 
            pass 
        else:
            assert False, "Ошибка спецификации модели: Campaign variables должен быть list or None"
        
        for x in model_spec['Campaign variables']:
            assert isinstance(x, dict), "Ошибка спецификации модели: Campaign variable должна быть dict " + str(x)
            assert list(x.keys()) == self.campaign_var_definition, "Ошибка спецификации модели: Campaign variable " + str(x)
            assert x['Name'] in model_data.columns, \
                "Ошибка спецификации модели: Campaign variable '" + x['Name'] + "' нет в данных"

        # проверка Other variables
        # НЕобязательная секция
        # может быть списком, может быть пустым, строкой или None
        if ('Other variables' not in model_spec) or (model_spec['Other variables'] is None):
            model_spec['Other variables'] = []
        elif isinstance(model_spec['Other variables'], str):
            model_spec['Other variables'] = [model_spec['Other variables']]
        elif isinstance(model_spec['Other variables'], list):
            pass
        else: 
            assert False, "Ошибка спецификации модели: Other variables неверный формат"

        for x in model_spec['Other variables']: 
            assert x in model_data.columns, "Ошибка спецификации модели: Other variable '" + x + "' нет в данных"

            
        # проверка RG variable
        if 'Relevance groups variable' not in model_spec: 
            model_spec['Relevance groups variable'] = None
        elif model_spec['Relevance groups variable'] is None:
            pass
        elif isinstance(model_spec['Relevance groups variable'], str): 
             assert model_spec['Relevance groups variable'] in model_data.columns, \
                "Ошибка спецификации модели: Relevance groups variable '" +\
                    model_spec['Relevance groups variable'] + "' нет в данных"
             assert model_spec['Relevance groups variable'] not in model_spec['Target variables'], \
                "Ошибка спецификации модели: одна и та же переменная в Relevance groups variable и Target variables. Так не работает"
        else: 
            assert False, """Ошибка спецификации модели: Relevance groups variable неверный формат. 
                             Принимается только ОДНА переменная. Может быть только строкой, None, или отсутсвовать"""
        
        
        # проверка Subsample variable
        # НЕобязательная секция
        # может быть списком, может быть пустым, строкой или None
        if ('Subsample variables' not in model_spec) or (model_spec['Subsample variables'] is None):
            model_spec['Subsample variables'] = []
        elif isinstance(model_spec['Subsample variables'], str):
            model_spec['Subsample variables'] = [model_spec['Subsample variables']]
        elif isinstance(model_spec['Subsample variables'], list):
            pass
        else: 
            assert False, "Ошибка спецификации модели: Subsample variables неверный формат"

        for x in model_spec['Subsample variables']: 
            assert x in model_data.columns, "Ошибка спецификации модели: Subsample variables '" + x + "' нет в данных"
        
        return model_spec
        
        
    def ContributionsOneTarget(self, model_data, target_var, model_spec):
        assert isinstance(target_var, str), "ContributionsOneTarget: target_var должно быть строкой"
        assert target_var in model_data.columns, "ContributionsOneTarget: target_var нет в датасете"
        
        if model_spec['Relevance groups variable']: 
            relevance_groups = model_spec['Relevance groups variable']
        else:
            relevance_groups = 'Total'
        assert target_var != relevance_groups, "ContributionsOneTarget: target_var и 'Relevance groups variable' совпадают. Так не работает."


        data = model_data.copy()  # копируем чтобы не сохранять мусор

        data['Base'] = 1   # intercept для регрессии
        data['Total'] = 'all'  # for reporting total 

        # HillsTransformation над CampaignVariables
        # пишем в те же столбцы
        for c in model_spec['Campaign variables']:
            data[c['Name']] = HillsTransformation(data[c['Name']], c['Alpha'], c['Gamma'])

        X_names = [x['Name'] for x in model_spec['Campaign variables']] + model_spec['Other variables'] + ['Base']
        contrib_names = [x + ' contribution' for x in X_names]

        for group in data[relevance_groups].unique():
            sub_sample = (data[relevance_groups]==group)
            X = np.array(data.loc[sub_sample, X_names])
            y = np.array(data.loc[sub_sample, target_var]).flatten()

            reg = LR(fit_intercept=False).fit(X, y)
            betas = reg.coef_.copy()
            betas[reg.p.flatten() > self.p_value] = 0 

            data.loc[sub_sample, contrib_names] = X * betas

        
        splits = model_spec['Subsample variables']
        split_reports = []
        split_names = []

        for split in ['Total'] + splits + ([] if relevance_groups == 'Total' else [relevance_groups]): 
            for group in data[split].unique():
                sub_sample = (data[split]==group)
                contribs = data.loc[sub_sample, [target_var] + contrib_names].mean()
                contribs.index = ['Observed value'] + X_names

                if self.correct_base:
                    contribs['Base'] += (contribs['Observed value'] - contribs[X_names].sum())

                split_reports.append(contribs)
                split_names.append(split + '==' + str(group))

        return pd.concat(split_reports, keys=split_names).rename(target_var, inplace=True)
    
    def ContributionsMultiTarget(self, model_data, model_spec):
        #return [self.ContributionsOneTarget(model_data, t, model_spec) for t in model_spec['Target variables']]
        
        return pd.concat(
            [self.ContributionsOneTarget(model_data, t, model_spec) for t in model_spec['Target variables']], 
            axis=1
        )
    
    
    def RunModelFromFilesSpec(self, data_file, spec_file, result_file=None): 
        model_data = pd.read_csv(data_file)

        with open(spec_file, 'r') as f: 
            model_spec = json.load(f)
        
        model_spec = self.CheckFixSpec(model_data, model_spec)
        
        result = self.ContributionsMultiTarget(model_data, model_spec)
        if result_file:
            result.to_excel(result_file + '.xlsx')
        else: 
            print(result)

    def ValidateNonCampaignVariables(self, model_data, target_var, model_spec): 
        assert isinstance(model_spec['Other variables'], list), "Ошибка спецификации: ожидается непустой список в other variables"
        assert len(model_spec['Other variables']) > 0, "Ошибка спецификации: ожидается непустой список в other variables"

        data = model_data.copy()  # копируем чтобы не сохранять мусор
        data['Base'] = 1   # intercept для регрессии
        
        X_names = ['Base']

        model_log = []
        R2_log = []

        model_log_index = pd.Index(
            ['Model', 'R2'] +\
            ['Beta ' + x for x in (['Base'] + model_spec['Other variables'])] +\
            ['P-value ' + x for x in (['Base'] + model_spec['Other variables'])]
        )

        while len(X_names) <= len(model_spec['Other variables']): 
            r2_max = 0
            next_candidate = ''
            for v in model_spec['Other variables']: 
                if v in X_names: 
                    continue
                X_names_this_model = X_names + [v]
                reg = LR(fit_intercept=False).fit(data[X_names_this_model], data[target_var])
                r2 = reg.score(data[X_names_this_model], data[target_var])
                if r2 > r2_max:
                    r2_max = r2
                    next_candidate = v
                    
                model_log.append(pd.Series(index=model_log_index))

                model_log[-1]['Model'] = str(X_names_this_model)
                model_log[-1]['R2'] = r2

                for var, beta, pvalue in zip(X_names_this_model, reg.coef_, reg.p):
                    model_log[-1]['Beta ' + var] = beta
                    model_log[-1]['P-value ' + var] = pvalue
            
            print(next_candidate, 'is taken with R2: ', r2_max)
            X_names.append(next_candidate)
            R2_log.append(r2_max)

        fig, ax = plt.subplots() 
        x_axis = range(1, len(model_spec['Other variables']) + 1)
        ax.plot(x_axis, R2_log)
        ax.set_xticks(x_axis)

        return pd.DataFrame(model_log)




        