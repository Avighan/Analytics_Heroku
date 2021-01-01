import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, pearsonr, spearmanr, kendalltau, chi2_contingency, ttest_ind,ttest_rel, f_oneway, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew


class Statistics:

    def __init__(self, stats_df):
        self.stats_df = stats_df

    def __get__stats__(self,col):
        return self.stats_df[col].value_counts()


    def normality_tests(self, data, test_type):
        # Tests whether a data sample has a Gaussian distribution.
        # H0: the sample has a Gaussian distribution.
        # H1: the sample does not have a Gaussian distribution
        if test_type == 'ShapiroWilk':
            stat, p = shapiro(data)
            if p > 0.05:
                return 'Probably Gaussian',p
            else:
                return 'Probably not Gaussian',p

        elif test_type == 'DAgostino':
            stat, p = normaltest(data)
            if p > 0.05:
                return 'Probably Gaussian',p
            else:
                return 'Probably not Gaussian',p

        elif test_type == 'AndersonDarling':
            result = anderson(data)
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic < cv:
                    return 'Probably Gaussian at the %.1f%% level' % (sl),cv
                else:
                    return 'Probably not Gaussian at the %.1f%% level' % (sl),cv

    def correlation_tests(self, data1, data2, test_type):
        # Tests whether two samples have a linear relationship.
        # Assumptions
        # Observations in each sample are independent and identically distributed (iid).
        # Observations in each sample are normally distributed.
        # Observations in each sample have the same variance.
        # H0: the two samples are independent.
        # H1: there is a dependency between the samples.
        if test_type == 'pearson':
            stat, p = pearsonr(data1, data2)
            if p > 0.05:
                return 'Probably independent'
            else:
                return 'Probably dependent'
        elif test_type == 'spearmans':
            stat, p = spearmanr(data1, data2)
            if p > 0.05:
                return 'Probably independent'
            else:
                return 'Probably dependent'
        elif test_type == 'kendalls':
            stat, p = kendalltau(data1, data2)
            if p > 0.05:
                return 'Probably independent'
            else:
                return 'Probably dependent'
        elif test_type == 'chi':
            table = pd.concat([data1, data2], axis=1)
            stat, p, dof, expected = chi2_contingency(table)
            if p > 0.05:
                return 'Probably independent'
            else:
                return 'Probably dependent'

    def stationary_tests(self, data, test_type):
        # Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive.
        # Observations in are temporally ordered.
        # H0: a unit root is present (series is non-stationary).
        # H1: a unit root is not present (series is stationary).
        if test_type == 'adfuller':
            stat, p, lags, obs, crit, t = adfuller(data)
            if p > 0.05:
                return 'Probably not Stationary'
            else:
                return 'Probably Stationary'
        elif test_type == 'kpss':
            ## Example of the Kwiatkowski-Phillips-Schmidt-Shin test
            stat, p, lags, crit = kpss(data)
            if p > 0.05:
                return 'Probably not Stationary'
            else:
                return 'Probably Stationary'

    def parametric_tests(self, data1, data2, test_type):
        # Tests whether the means of two independent samples are significantly different.
        # Observations in each sample are independent and identically distributed (iid).
        # Observations in each sample are normally distributed.
        # Observations in each sample have the same variance.
        # H0: the means of the samples are equal.
        # H1: the means of the samples are unequal.
        if test_type == 'studentttest':
            stat, p = ttest_ind(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'
        elif test_type == 'pairedttest':
            stat, p = ttest_rel(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'
        elif test_type == 'anova':
            stat, p = f_oneway(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'

    def non_parametric_tests(self, data1, data2, test_type):
        # Tests whether the distributions of two independent samples are equal or not.
        # Observations in each sample are independent and identically distributed (iid).
        # Observations in each sample can be ranked.
        # H0: the distributions of both samples are equal.
        # H1: the distributions of both samples are not equal.
        if test_type == 'mannwhitneyu':
            stat, p = mannwhitneyu(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'
        elif test_type == 'wilcoxon':
            stat, p = wilcoxon(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'
        elif test_type == 'kruskal':
            stat, p = kruskal(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'
        elif test_type == 'friedmanchisquare':
            stat, p = friedmanchisquare(data1, data2)
            if p > 0.05:
                return 'Probably the same distribution'
            else:
                return 'Probably different distributions'

    def run_one_sample_t_tests(self, df, test_col, compare_mean):
        # you have test_col value and you are checking whether test_col value is compare_mean or not.
        from scipy.stats import ttest_1samp
        import numpy as np
        col_mean = np.mean(df[test_col])
        tset, pval = ttest_1samp(col_mean, compare_mean)
        print("p - values", pval)
        if pval < 0.05:  # alpha value is 0.05 or 5%
            return " we are rejecting null hypothesis"
        else:
            return "we are accepting null hypothesis"

    def run_two_sample_t_tests(self, df, test_col1, test_col2):
        from scipy.stats import ttest_ind
        # Example : is there any association between test_col1 and test_col2
        col1_mean = np.mean(df[test_col1])
        col2_mean = np.mean(df[test_col2])
        col1_std = np.std(df[test_col1])
        col2_std = np.std(df[test_col2])
        ttest, pval = ttest_ind(test_col1, test_col2)
        if pval < 0.05:
            return "we reject null hypothesis"
        else:
            return "we accept null hypothesis"

    def paired_sampled_t_test(self, df, x_before, x_after):
        from scipy import stats
        # H0 :- means difference between two sample is 0
        # H1:- mean difference between two sample is not 0
        df[[x_before, x_after]].describe()
        ttest, pval = stats.ttest_rel(df[x_before], df[x_after])
        if pval < 0.05:
            return "reject null hypothesis"
        else:
            return "accept null hypothesis"

    # Conditions
    """
    Your sample size is greater than 30. Otherwise, use a t test.
    Data points should be independent from each other. In other words, one data point isn’t related or doesn’t affect another data point.
    Your data should be normally distributed. However, for large sample sizes (over 30) this doesn’t always matter.
    Your data should be randomly selected from a population, where each item has an equal chance of being selected.
    Sample sizes should be equal if at all possible.
    """

    def one_sampled_z_test(self, df, x_before, compare_mean):
        # we are using z-test for x_before with some mean like compare_mean
        from statsmodels.stats import weightstats as stests
        ztest, pval = stests.ztest(df['bp_before'], x2=None, value=compare_mean)
        if pval < 0.05:
            return "reject null hypothesis"
        else:
            return "accept null hypothesis"

    def two_sampled_z_test(self, df, x_before, x_after):
        from statsmodels.stats import weightstats as stests
        # H0 : mean of two group is 0
        # H1 : mean of two group is not 0
        # we are checking in x_before,x_after columns after and before situation.
        ztest, pval1 = stests.ztest(df[x_before], x2=df[x_after], value=0, alternative='two-sided')
        if pval1 < 0.05:
            return "reject null hypothesis"
        else:
            return "accept null hypothesis"

    def one_way_anova_test(self, df_anova, col_name1, colname2):
        # F = Between group variability / Within group variability
        # Unlike the z and t-distributions, the F-distribution does not have any negative values because between and within-group
        # variability are always positive due to squaring each deviation.
        # It tell whether two or more groups are similar or not based on their mean similarity and f-score.
        # there are different category of plant and their weight and need to check whether all 3 group are similar or not
        df_anova = df_anova[[col_name1, colname2]]
        grps = pd.unique(df_anova[col_name1].values)
        d_data = {grp: df_anova[colname2][df_anova[col_name1] == grp] for grp in grps}
        F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])
        if p < 0.05:
            return "reject null hypothesis"
        else:
            return "accept null hypothesis"

    def chi_sq_test(self,table):
        from scipy.stats import chi2_contingency
        from scipy.stats import chi2
        stat, p, dof, expected = chi2_contingency(table)
        print('dof=%d' % dof)
        print(expected)
        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        return 'probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat)
        if abs(stat) >= critical:
            return 'Dependent (reject H0)'
        else:
            return 'Independent (fail to reject H0)'
        # interpret p-value
        alpha = 1.0 - prob
        return 'significance=%.3f, p=%.3f' % (alpha, p)
        if p <= alpha:
            return 'Dependent (reject H0)'
        else:
            return 'Independent (fail to reject H0)'



    def check_skewness(self,x):
        skewness = skew(x)
        if skewness == 0:
            return 'Normally distributed'
        elif skewness > 0 :
            return 'right skewed distribution'
        else:
            return 'left skewed distribution'