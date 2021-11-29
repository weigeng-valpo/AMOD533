# import all packages
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
# set display options
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',200)
pd.set_option('display.width',200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
kwargs = dict(skipinitialspace=False,engine='c',encoding='ISO-8859-1')


import gc

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix,classification_report




class feature_explorer:
    
    def __init__(self,
                 X_vars,
                 y_vars,
                 fdf,
                 weights,
                 name='Model Name',
                 n=20,
                 null_value = 0,
                 use_FI = True,
                 use_IV = True,
                 use_VIF = True,
                 use_sig = False,
                 use_complete_targets = False,
                 balance_classes = True,
                 composite=True,
                 n_samples_feature_id = 10000,
                 frac_training = 1,
                 qu = 0.5,
                 ql = 0.5
                ):
        
        # Initialize
        self.name = name
        self.X_vars = X_vars
        self.y_vars = y_vars
        self.null_value = null_value
        self.use_FI = use_FI 
        self.use_IV = use_IV 
        self.use_VIF = use_VIF
        self.use_sig = use_sig
        self.n = n
        cols = X_vars
        cols.extend(y_vars)
        
        # Trim the dataset
        fdf = fdf[cols]
        
        # Just keep the rows where at least one target is filled
        fdf.dropna(subset=y_vars,thresh=1,inplace=True)
        
    
        # Identify the columns to dummify
        cols_to_dummy = list(fdf.select_dtypes(include=['object']).columns)
        listing = []
        for item in cols_to_dummy:
            if (fdf[item].nunique()) >= 6:
                fdf.drop([item],axis=1,inplace=True)
            else:
                listing.append(item)
        cols_to_dummy = listing
        
        # More initializations
        self.cols_to_dummy = cols_to_dummy
        
        # Get the dummies
        fdf = pd.get_dummies(fdf,drop_first=True,columns=cols_to_dummy)
        
        # Decide which rows to drop
        if use_complete_targets:
            fdf.dropna(subset=y_vars,inplace=True)
        
        # Compute the composite target for completed members
        for item in y_vars:
            if weights[item] != 0.0:
                fdf[f'{item}_denoms'] = fdf[item].apply(lambda x: 1 if float(x) in [0.0,1.0] else 0.0)
            else:
                fdf[f'{item}_denoms'] = 0.0
            fdf[f'{item}_mweights'] = weights[item]
            cols_for_target = list(fdf.filter(regex=item).columns)
            fdf[f'{item}_scores'] = fdf[cols_for_target].apply(lambda row: 0 if row[f'{item}_denoms'] == 0 else row[f'{item}_mweights'] * row[item] / row[f'{item}_denoms'],axis=1)
        cols_to_sum = list(fdf.filter(regex='_scores').columns)
        weight_cols = list(fdf.filter(regex='_mweights'))
        fdf['COMPOSITE_TARGET'] = fdf[cols_to_sum].sum(axis=1) / fdf[weight_cols].sum(axis=1)
        
        # Drop the columns that were created for the scoring and the initial target_cols
        fdf.drop(fdf.filter(regex='_denoms|_mweights|_scores'),axis=1,inplace=True)
        fdf.drop(y_vars,axis=1,inplace=True)
        
        # Generate HIST of composite target
        self.hist = fdf['COMPOSITE_TARGET'].hist();
        self.tar_val_counts = fdf['COMPOSITE_TARGET'].value_counts()
        self.tar_describe = fdf['COMPOSITE_TARGET'].describe()
        
        # Decide how to balance the binary target
        uq = fdf['COMPOSITE_TARGET'].quantile(qu)
        lq = fdf['COMPOSITE_TARGET'].quantile(ql)
        self.qu = uq
        self.ql = lq 
        if fdf['COMPOSITE_TARGET'].nunique() > 2:
            fdf['b_COMPOSITE_TARGET'] = fdf['COMPOSITE_TARGET'].apply(lambda x: 1 if x > uq else 0 if x <= lq else np.nan)        
        else:
            fdf ['b_COMPOSITE_TARGET'] = fdf['COMPOSITE_TARGET']
        
        # Generate HIST of binarized composite target
        # self.b_hist = fdf['b_COMPOSITE_TARGET'].hist()
        self.b_tar_val_counts = fdf['b_COMPOSITE_TARGET'].value_counts()
        
        
        # Decide on the feature selection dataset size
        if balance_classes:
            num_per_class = fdf['b_COMPOSITE_TARGET'].value_counts().min()
            fdf_fs = fdf.sample(frac=1,random_state=42).groupby('b_COMPOSITE_TARGET').head(n_samples_feature_id/2)
            fdf_m = fdf.sample(frac=1,random_state=42).groupby('b_COMPOSITE_TARGET').head(num_per_class)
        else:
            fdf_fs = fdf.sample(n_sample_feature_id)
            fdf_m = fdf.copy()
        
        
        
        
        # Decide on the modelling dataset size
        
        
        # Scale the dataset
        def scale_dataset(d,v):
            d.fillna(null_value,inplace=True)
            cols = d.columns
            scalar = MinMaxScaler()
            d = pd.DataFrame(scalar.fit_transform(d),columns=cols)
            return d
        
        fdf_m = scale_dataset(fdf_m,null_value)
        fdf_fs = scale_dataset(fdf_fs,null_value)
        
        #print(fdf_fs['COMPOSITE_TARGET'].value_counts())
        #print(fdf_fs['b_COMPOSITE_TARGET'].value_counts())
        # Drop the non-binarized composite target
        fdf_fs.drop(['COMPOSITE_TARGET'],axis=1,inplace=True)
        fdf.drop(['COMPOSITE_TARGET'],axis=1,inplace=True)
        
        
        
        self.df_fs = fdf_fs
        self.df_m = fdf_m
        self.df = fdf
        
    # Perform Feature selection
    def perform_feature_selection(self):
        if self.use_sig:
            n = self.n
            # Set up X and y
            d = self.df_fs
            X = d.drop(['b_COMPOSITE_TARGET'],axis=1)
            y = d['b_COMPOSITE_TARGET']

            # Get setup for the first iteration
            from xgboost import XGBClassifier
            import statsmodels.api as sm

            # Get IV from the IV file
            # from IV import data_vars
            # _,ivs = data_vars(X,y)
            # ivs.set_index('VAR_NAME',inplace=True)

            # Calculate initial variance inflation factor
            from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
            X['constant'] = [1] * len(X)
            X_drop_extras = X.dropna(axis='columns')
            vif_data = pd.DataFrame()
            vif_data['Variable'] = X_drop_extras.columns
            vif_data["VIF"] = [vif(X_drop_extras.values,i) for i in range(len(X_drop_extras.columns))]
            vif_data.set_index('Variable',inplace = True)
            X.drop(['constant'],axis=1,inplace=True)

            # Set up an XGBoost classifier to get initial feature importances
            clf = XGBClassifier(random_state = 42,
                           n_jobs=-1,
                           max_depth=3,
                           n_estimators = 400,
                           learning_rate = 0.1,
                           min_child_weight = 4)
            clf.fit(X,y,eval_metric='error')

            # Combine these two measures
            # Compile these three measures into one frame for ranking
            dfr = pd.DataFrame(index = d.drop(['b_COMPOSITE_TARGET'],axis=1).columns)
            dfr['FI'] = clf.feature_importances_
            dfr['FI - rank'] = dfr['FI'].rank(ascending=False)
            # dfr = pd.merge(dfr,ivs,how='left',left_index=True,right_index=True)
            # dfr['IV - rank'] = dfr['IV'].rank(ascending=False)
            dfr = pd.merge(dfr,vif_data,how='left',left_index=True,right_index=True)
            dfr['VIF - rank'] = dfr['VIF'].rank(ascending=True)



            dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'])/2

            dftemp = dfr[dfr['VIF'] > 20]
            qut = dftemp['VIF - rank'].quantile(.45)
            dftemp = dftemp[dftemp['VIF - rank'] >= qut]
            cols_to_cut = list(dftemp.index)

            cols_to_drop = list(dfr[dfr['VIF'].isnull()].index)
            cols_to_drop.extend(cols_to_cut)
            cols_to_drop_c = cols_to_drop

    #         dfr.drop(cols_to_drop,inplace=True)
    #         display(dfr)

    #         dfr['r1'] = (dfr['FI - rank'] +  2 * dfr['VIF - rank'])/3

            cols_to_keep = dfr[(dfr['FI'] >= 0.01) & (dfr['VIF'] <= 20)]
            cols_to_keep = [i for i in list(dfr.columns) if i in cols_to_keep]
            more_to_drop = [i for i in list(dfr.columns)if i not in cols_to_keep]
            cols_to_drop.extend(more_to_drop)
            dfr.drop(cols_to_drop,inplace=True)
            X.drop(cols_to_drop,axis=1,inplace=True)

                    # Set up an Logit regressor to find significance
            # Train a logistic classifier to get the coeff and sig
            df_test = pd.DataFrame(X)
            df_test['target'] = y
    #         display(df_test.corr())
    #         df_test.corr().to_clipboard()
    #         df_test.to_clipboard()
            log_reg = sm.Logit(y,X).fit(disp=0)
            dfr['LR_Coeff'] = log_reg.params
            dfr['LR_Sig'] = log_reg.pvalues
            dfr['LR_Sig - rank'] = dfr['LR_Sig'].rank(ascending=True)

            dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'] + dfr['LR_Sig - rank'])/3

            while len(dfr) > n:
    #             print(len(dfr))
                dfr = dfr.sort_values('r1',ascending=False)
                dfr = dfr.iloc[1:,:]
                X = X[list(dfr.index)]
                cols_to_drop = []
                cols = list(X.columns)

                dft = dfr.copy()

                # drop columns where both IV and FI are low from dft
                if len(dft[(dft['FI']  < 0.002)&(dft['LR_Sig'] > 0.2)]) > 0:
                    dft = dft[~((dft['FI']  < 0.002)&(dft['FI'] < 0.002))]
                    cols_to_drop = [i for i in cols if i not in list(dft.index)]

                # Drop columns with low importance or low value
                X.drop(cols_to_drop,axis=1,inplace=True)

                # Retrain a classifier to get new FI
                clf = XGBClassifier(random_state = 42,
                           n_jobs=-1,
                           max_depth=3,
                           n_estimators = 400,
                           learning_rate = 0.1,
                           min_child_weight = 4)
                clf.fit(X,y,eval_metric='error');
                dfr = pd.DataFrame(index = X.columns)
                dfr['FI'] = clf.feature_importances_

                # Train a logistic classifier to get the coeff and sig
                log_reg = sm.Logit(y,X).fit(disp=0);
                dfr['LR_Coeff'] = log_reg.params
                dfr['LR_Sig'] = log_reg.pvalues
                dfr['LR_Sig - rank'] = dfr['LR_Sig'].rank(ascending=True)

                # Get new VIF
                X['constant'] = [1] * len(X)
                X_drop_extras = X.dropna(axis='columns')
                vif_data = pd.DataFrame()
                vif_data['Variable'] = X_drop_extras.columns
                vif_data["VIF"] = [vif(X_drop_extras.values,i) for i in range(len(X_drop_extras.columns))]
                vif_data.set_index('Variable',inplace = True)
                X.drop(['constant'],axis=1,inplace=True)

                # Merge the three metrics into one ranking
                dfr['FI - rank'] = dfr['FI'].rank(ascending=False)
    #             dfr = pd.merge(dfr,ivs,how='left',left_index=True,right_index=True)
    #             dfr['IV - rank'] = dfr['IV'].rank(ascending=False)
                dfr = pd.merge(dfr,vif_data,how='left',left_index=True,right_index=True)
                dfr['VIF - rank'] = dfr['VIF'].rank(ascending=True)
                dfr['SIG - rank'] = dfr['LR_Sig'].rank(ascending=True)

                dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'] + dfr['LR_Sig - rank'])/3




            self.dfr = dfr
            self.sig_vars = dfr[dfr['LR_Sig'] <= 0.05]
            self.dropped_cols = cols_to_drop
            self.X = X
            self.y = y
            
        else:
            n = self.n
            # Set up X and y
            d = self.df_fs
            X = d.drop(['b_COMPOSITE_TARGET'],axis=1)
            y = d['b_COMPOSITE_TARGET']

            # Get setup for the first iteration
            from xgboost import XGBClassifier
            import statsmodels.api as sm

            # Get IV from the IV file
            # from IV import data_vars
            # _,ivs = data_vars(X,y)
            # ivs.set_index('VAR_NAME',inplace=True)

            # Calculate initial variance inflation factor
            from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
            X['constant'] = [1] * len(X)
            X_drop_extras = X.dropna(axis='columns')
            vif_data = pd.DataFrame()
            vif_data['Variable'] = X_drop_extras.columns
            vif_data["VIF"] = [vif(X_drop_extras.values,i) for i in range(len(X_drop_extras.columns))]
            vif_data.set_index('Variable',inplace = True)
            X.drop(['constant'],axis=1,inplace=True)

            # Set up an XGBoost classifier to get initial feature importances
            clf = XGBClassifier(random_state = 42,
                           n_jobs=-1,
                           max_depth=3,
                           n_estimators = 400,
                           learning_rate = 0.1,
                           min_child_weight = 4)
            clf.fit(X,y,eval_metric='error')

            # Combine these two measures
            # Compile these three measures into one frame for ranking
            dfr = pd.DataFrame(index = d.drop(['b_COMPOSITE_TARGET'],axis=1).columns)
            dfr['FI'] = clf.feature_importances_
            dfr['FI - rank'] = dfr['FI'].rank(ascending=False)
            # dfr = pd.merge(dfr,ivs,how='left',left_index=True,right_index=True)
            # dfr['IV - rank'] = dfr['IV'].rank(ascending=False)
            dfr = pd.merge(dfr,vif_data,how='left',left_index=True,right_index=True)
            dfr['VIF - rank'] = dfr['VIF'].rank(ascending=True)



            dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'])/2

            dftemp = dfr[dfr['VIF'] > 20]
            qut = dftemp['VIF - rank'].quantile(.45)
            dftemp = dftemp[dftemp['VIF - rank'] >= qut]
            cols_to_cut = list(dftemp.index)

            cols_to_drop = list(dfr[dfr['VIF'].isnull()].index)
            cols_to_drop.extend(cols_to_cut)
            cols_to_drop_c = cols_to_drop

    #         dfr.drop(cols_to_drop,inplace=True)
    #         display(dfr)

    #         dfr['r1'] = (dfr['FI - rank'] +  2 * dfr['VIF - rank'])/3

            cols_to_keep = dfr[(dfr['FI'] >= 0.01) & (dfr['VIF'] <= 20)]
            cols_to_keep = [i for i in list(dfr.columns) if i in cols_to_keep]
            more_to_drop = [i for i in list(dfr.columns)if i not in cols_to_keep]
            cols_to_drop.extend(more_to_drop)
            dfr.drop(cols_to_drop,inplace=True)
            X.drop(cols_to_drop,axis=1,inplace=True)

                    # Set up an Logit regressor to find significance
            # Train a logistic classifier to get the coeff and sig
            df_test = pd.DataFrame(X)
            df_test['target'] = y
    #         display(df_test.corr())
    #         df_test.corr().to_clipboard()
    #         df_test.to_clipboard()
#             log_reg = sm.Logit(y,X).fit(disp=0)
#             dfr['LR_Coeff'] = log_reg.params
#             dfr['LR_Sig'] = log_reg.pvalues
#             dfr['LR_Sig - rank'] = dfr['LR_Sig'].rank(ascending=True)

#             dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'] + dfr['LR_Sig - rank'])/3
            dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'])/2

            while len(dfr) > n:
                last_run = False
    #             print(len(dfr))
                if len(dfr) <= n+1:
                    last_run = True
                dfr = dfr.sort_values('r1',ascending=False)
                dfr = dfr.iloc[1:,:]
                X = X[list(dfr.index)]
                cols_to_drop = []
                cols = list(X.columns)

                dft = dfr.copy()

                # drop columns where both IV and FI are low from dft
#                 if len(dft[(dft['FI']  < 0.002)&(dft['LR_Sig'] > 0.2)]) > 0:
#                     dft = dft[~((dft['FI']  < 0.002)&(dft['FI'] < 0.002))]
#                     cols_to_drop = [i for i in cols if i not in list(dft.index)]

                # Drop columns with low importance or low value
#                 X.drop(cols_to_drop,axis=1,inplace=True)

                # Retrain a classifier to get new FI
                clf = XGBClassifier(random_state = 42,
                           n_jobs=-1,
                           max_depth=3,
                           n_estimators = 400,
                           learning_rate = 0.1,
                           min_child_weight = 4)
                clf.fit(X,y,eval_metric='error');
                dfr = pd.DataFrame(index = X.columns)
                dfr['FI'] = clf.feature_importances_

                # Train a logistic classifier to get the coeff and sig
#                 log_reg = sm.Logit(y,X).fit(disp=0);
#                 dfr['LR_Coeff'] = log_reg.params
#                 dfr['LR_Sig'] = log_reg.pvalues
#                 dfr['LR_Sig - rank'] = dfr['LR_Sig'].rank(ascending=True)

                # Get new VIF
                X['constant'] = [1] * len(X)
                X_drop_extras = X.dropna(axis='columns')
                vif_data = pd.DataFrame()
                vif_data['Variable'] = X_drop_extras.columns
                vif_data["VIF"] = [vif(X_drop_extras.values,i) for i in range(len(X_drop_extras.columns))]
                vif_data.set_index('Variable',inplace = True)
                X.drop(['constant'],axis=1,inplace=True)

                # Merge the three metrics into one ranking
                dfr['FI - rank'] = dfr['FI'].rank(ascending=False)
    #             dfr = pd.merge(dfr,ivs,how='left',left_index=True,right_index=True)
    #             dfr['IV - rank'] = dfr['IV'].rank(ascending=False)
                dfr = pd.merge(dfr,vif_data,how='left',left_index=True,right_index=True)
                dfr['VIF - rank'] = dfr['VIF'].rank(ascending=True)
#                 dfr['SIG - rank'] = dfr['LR_Sig'].rank(ascending=True)

#                 dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'] + dfr['LR_Sig - rank'])/3
                dfr['r1'] = (dfr['FI - rank'] + dfr['VIF - rank'])/2

                if last_run: 
                    log_reg = sm.Logit(y,X).fit(disp=0)
                    dfr['LR_Coeff'] = log_reg.params
                    dfr['LR_Sig'] = log_reg.pvalues
                    dfr['LR_Sig - rank'] = dfr['LR_Sig'].rank(ascending=True)


            self.dfr = dfr
            self.sig_vars = dfr[dfr['LR_Sig'] <= 0.05].sort_values('LR_Coeff',ascending=True)
            self.dropped_cols = cols_to_drop
            self.X = X
            self.y = y
        
    # Train model
    def train_model(self):
        target_col = 'b_COMPOSITE_TARGET'
        trial1 = self.df_m.copy()
        cols = list(self.X.columns)
        cols1 = [i for i in cols]
        cols.append(target_col)
        trial1 = trial1[cols]
        X = self.X.values
        y = self.y.values

        n_samples,n_features = X.shape
        random_state = np.random.RandomState(42)
        cv = StratifiedKFold(n_splits=5)
        clf = XGBClassifier(random_state = 42,
                           n_jobs=-1,
                           max_depth=3,
                           n_estimators = 400,
                           learning_rate = 0.1,
                           min_child_weight = 4)

        tprs,aucs = [],[]
        mean_fpr = np.linspace(0,1,100)

        fig,ax = plt.subplots()

        for i, (train,test) in enumerate(cv.split(X,y)):
            clf.fit(X[train],y[train],eval_metric='error')
            viz = plot_roc_curve(clf,X[test],y[test],name = 'ROC fold {}'.format(i),alpha=0.3,lw=1,ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        plt.show()
        X_full = trial1[cols1].values
        self.X_full = X_full
        y_full = trial1[target_col]
        self.y_full = y_full
        y_true = y_full
        y_pred = clf.predict(X_full)
        tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
        self.class_summary = f'True Negative:  {tn}\nFalse Positive: {fp}\nFalse Negative: {fn}\nTrue Positive:  {tp}'
        self.class_report = classification_report(y_true,y_pred)
        self.confusion_matrix = confusion_matrix(y_true,y_pred)