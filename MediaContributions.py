import numpy as np 
import pandas as pd

import json

from LinearRegression import LR, SubSampleRegression
from HillsTransformation import HillsTransformation


import matplotlib.pyplot as plt
import seaborn as sns

def AppendSeriesNameToItsValues(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: col.name + ' == ' + str(x))

class ModelSpec: 
    def __init__(self, model_spec: list): 
        pass
    


class MediaContributionCalculator: 
    obligatory_sections = ['Target variables', 'Campaign variables']
    campaign_var_definition = ['Name', 'Alpha', 'Gamma']
    campaign_var_suffix = '_dr'
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
        elif isinstance(model_spec['Relevance groups variable'], list):
             assert all([v in model_data.columns for v in model_spec['Relevance groups variable']]), \
                "Ошибка спецификации модели:  не все Relevance groups variable есть в данных"""
             ######## проверка что нет среди таргетов 
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
        

    def __ContributionsOneTarget(self, model_data: pd.DataFrame, model_spec: list, target_var: str) -> pd.DataFrame:
        assert (model_spec['Relevance groups variable'] is None) or isinstance(model_spec['Relevance groups variable'], str), \
            "Relevance groups variable здесь может быть или строкой или None, НЕ списком"
        
        # копируем чтобы не сохранять мусор 
        data = model_data.copy()
        # diminishing return transformation
        campaign_vars = self.__TransformCampaignVars(data, model_spec)
        # intercept
        data['Base'] = 1
        
        X_names = campaign_vars + model_spec['Other variables'] + ['Base']
        model = self.__FitOneModel(data, X_names, target_var, model_spec['Relevance groups variable'], self.p_value)
        contribs = model.Contributions(data) 
        # добавляем исходную переменную для потсроения таблиц
        contribs['Observed value'] = data[target_var]
        
        # for reporting total sample
        data['Total'] = 'all'
        if model_spec['Relevance groups variable'] and model_spec['Relevance groups variable'] not in model_spec['Subsample variables']:
            rg_split = [model_spec['Relevance groups variable']]
        else:
            rg_split = []

        reporting_splits = ['Total'] + rg_split + model_spec['Subsample variables']
        reporting = []
        for rs in reporting_splits: 
            report = contribs.groupby(AppendSeriesNameToItsValues(data[rs])).mean()
            report['Base'] -= (report.sum(axis=1) - 2 * report['Observed value'])
            reporting.append(report.T.unstack())

        #return contribs.groupby(AppendSeriesNameToItsValues(data[model_spec['Subsample variables'][0]])).mean()
        return pd.concat(reporting).rename(target_var, inplace=True)


    def GetContributions(self, model_data: pd.DataFrame, model_spec: list) -> pd.DataFrame: 
        return pd.concat(
            [self.__ContributionsOneTarget(model_data, model_spec, t) for t in model_spec['Target variables']], 
            axis=1 
        )
    

        
    def ContributionsOneTarget_(self, model_data, target_var, model_spec):
        
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
    


    def ContributionsMultiTarget_OLD(self, model_data, model_spec):
        # Старая версия, пока оставим для двойного теста
        return pd.concat(
            [self.ContributionsOneTarget_(model_data, t, model_spec) for t in model_spec['Target variables']], 
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


    def __ValidateNonCampaignVariables_OneTarget(self, model_data: pd.DataFrame, model_spec: list, target_var: str) -> pd.DataFrame:
        assert isinstance(model_spec['Other variables'], list), "Ошибка спецификации: ожидается непустой список в other variables"
        assert len(model_spec['Other variables']) > 0, "Ошибка спецификации: ожидается непустой список в other variables" 
        assert (model_spec['Relevance groups variable'] is None) or isinstance(model_spec['Relevance groups variable'], str), \
            "Ошибка спецификации: ожидается None или строка в Relevance groups variable"

        data = model_data.copy()
        campaign_vars = self.__TransformCampaignVars(data, model_spec)
        data['Base'] = 1

        X_names = ['Base']

        log_ = []
        r2_log_ = []
        while len(X_names) <= len(model_spec['Other variables']): 
            r2_max = 0
            next_candidate = ''
            for v in model_spec['Other variables']: 
                if v in X_names: 
                    continue
                X_names_for_this_model = X_names + [v]
                model = self.__FitOneModel(data, X_names_for_this_model, target_var, model_spec['Relevance groups variable'], 1)
                r2 = model.Score(data, target_var)
                if r2 > r2_max:
                    r2_max = r2
                    next_candidate = v

                betas = model.GetBetas()
                record_index = pd.MultiIndex.from_arrays(
                    [   [str(X_names_for_this_model) + '->' + target_var] * len(betas), 
                        [round(r2, 3)] * len(betas), 
                        betas.index
                    ], names=['Model', 'R2', 'RG values']
                ) 
                    
                log_.append(betas.set_index(record_index))
            
            #print(next_candidate, 'is taken with R2: ', r2_max)
            X_names.append(next_candidate)
            r2_log_.append(r2_max)
        
        fig, ax = plt.subplots() 
        x_axis = range(1, len(model_spec['Other variables']) + 1)
        ax.plot(x_axis, r2_log_)
        ax.set_title(target_var)
        ax.set_xticks(x_axis)
        
        return pd.concat(log_)

    
    def ValidateNonCampaignVariables(self, model_data: pd.DataFrame, model_spec: list) -> pd.DataFrame:
        return pd.concat(
            [self.__ValidateNonCampaignVariables_OneTarget(model_data, model_spec, t) for t in model_spec['Target variables']]
        )

    
    def __TransformCampaignVars(self, model_data, model_spec):
        camp_vars = []
        for c in model_spec['Campaign variables']:
            model_data[c['Name']+self.campaign_var_suffix] = HillsTransformation(model_data[c['Name']], c['Alpha'], c['Gamma'])
            camp_vars.append(c['Name']+self.campaign_var_suffix)
        return camp_vars
    
    def __FitOneModel(self, data: pd.DataFrame,  X_names: list, y_name: str, RG_name: str=None, p_value: float =0.05) -> SubSampleRegression: 
        return SubSampleRegression(p_value).Fit(data, X_names, y_name, RG_name)
        
    
    def ValidateRGVariables(self, model_data: pd.DataFrame, model_spec: dict) -> pd.DataFrame: 
        assert(isinstance(model_spec['Relevance groups variable'], list)), "Ожидаем list в Relevance groups variable"
        
        data = model_data.copy()
        campaign_vars = self.__TransformCampaignVars(data, model_spec)
        data['Base'] = 1
        
        log_ = []
        X_names = campaign_vars + model_spec['Other variables'] + ['Base']
        for rg_var in model_spec['Relevance groups variable']:
            for target_var in model_spec['Target variables']: 
                model = self.__FitOneModel(data, X_names, target_var, rg_var, 1)
                r2 = round(model.Score(data, target_var), 3)
                betas = model.GetBetas()
                index_arr = [
                    [rg_var + '->' + target_var] * len(betas), 
                    [r2] * len(betas), 
                    betas.index 
                ]
                record_index = pd.MultiIndex.from_arrays(index_arr, names=['RG var -> target var', 'R2', 'RG values'])
                log_.append(
                    betas.set_index(record_index)
                )

        return pd.concat(log_)
    

    def __ExploreDiminishingReturn__OneTarget(self, model_data: pd.DataFrame, model_spec: list, target_var: str, camp_var: str) -> pd.DataFrame:
        assert (model_spec['Relevance groups variable'] is None) or isinstance(model_spec['Relevance groups variable'], str), \
            "В этом тесте Relevance groups variable должно быть None или строкой"
        
        alpha_range = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16]
        gamma_range = [1, 2, 4, 6, 8, 10, 12, 14]

        fit_results = pd.DataFrame(index=alpha_range, columns=gamma_range, dtype='float')
        
        data = model_data.copy()
        data['Base'] = 1 

        X_names = model_spec['Other variables'] + ['Base']

        for alpha in alpha_range:
            for gamma in gamma_range: 
                data[camp_var + '_dr_transformed'] = HillsTransformation(data[camp_var], alpha, gamma) 
                X_names_for_this_model = [camp_var + '_dr_transformed'] + X_names
                model = self.__FitOneModel(data, X_names_for_this_model, target_var, model_spec['Relevance groups variable'], 1)
                fit_results.loc[alpha, gamma] = model.Score(data, target_var)
                
        fit_results.index.set_names('Alpha', inplace=True)
        fit_results.columns.set_names('Gamma', inplace=True)
        
        position_of_max = np.unravel_index(np.argmax(fit_results), fit_results.shape) 
        alpha_best = fit_results.index[position_of_max[0]]
        gamma_best = fit_results.columns[position_of_max[1]]
        print('For model {}->{}, best alpha: {}, best gamma: {}'.format(camp_var, target_var, alpha_best, gamma_best))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        sns.heatmap(fit_results, annot=True, ax=ax1)
        
        x_line = np.arange(0, 30, dtype='double')
        ax2.plot(x_line, HillsTransformation(x_line, alpha_best, gamma_best), 
                label='alpha:' + str(alpha_best) + ' gamma:' + str(gamma_best))
        ax2.legend()
        
        plt.show()
        
        return fit_results.set_index(
            pd.MultiIndex.from_arrays([[camp_var + '->' + target_var] * len(fit_results), fit_results.index])
        )
    
    def __ExploreDiminishingReturn__MultiTarget(self, model_data: pd.DataFrame, model_spec: list, camp_var: str) -> pd.DataFrame:
        return pd.concat(
            [self.__ExploreDiminishingReturn__OneTarget(model_data, model_spec, t, camp_var) for t in model_spec['Target variables']]
        )
    
    def ExploreDiminishingReturn(self, model_data: pd.DataFrame, model_spec: list) -> pd.DataFrame:
        return pd.concat(
            [self.__ExploreDiminishingReturn__MultiTarget(model_data, model_spec, c['Name']) for c in model_spec['Campaign variables']]
        )






        