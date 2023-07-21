import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from scipy.stats import sem


class PlotResults:
    def __init__(self,
                 classifiers_names,
                 sample_size,
                 label_size=26,
                 font_size=24,
                 ax_width=0.3,
                 bottom_size=0.3,
                 label1='Literature',
                 label2='Entropy-Complexty',
                 label3='Literature + Entropy-Complexty',
                 output_folder='.'):
        self.classifiers_names = classifiers_names
        self.sample_size = sample_size
        self.label_size = label_size
        self.font_size = font_size
        self.ax_width = ax_width
        self.bottom_size = bottom_size
        self.color1 = '#191970'
        self.color2 = '#6495ed'
        self.color3 = '#0000ff'
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3
        self.__output_folder = output_folder

    @staticmethod
    def __autolabel(rects, ax, vertical_pos=.5):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., vertical_pos,
                    f'{height:3.2}', fontsize=15,
                    ha='center', va='bottom')

    def plot_accuracy(self,
                      y_lt_accuracy, y_lt_std_error_accuracy,
                      y_inf_accuracy, y_inf_std_error_accuracy,
                      y_both_accuracy, y_both_std_error_accuracy):
        X_axis = np.arange(len(self.classifiers_names))
        fig, ax = plt.subplots(1, 1, figsize=(17, 9))

        # Classifier's Accuracy
        bars_literature = ax.bar(X_axis - self.ax_width, y_lt_accuracy, self.ax_width,
                                 label=self.label1,
                                 yerr=y_lt_std_error_accuracy, color=self.color1)
        bars_inf_teory = ax.bar(X_axis, y_inf_accuracy, self.ax_width,
                                label=self.label2,
                                yerr=y_inf_std_error_accuracy, color=self.color2)
        bars_both = ax.bar(X_axis + self.ax_width, y_both_accuracy, self.ax_width,
                           label=self.label3,
                           yerr=y_both_std_error_accuracy, color=self.color3)
        ax.set_xticks(X_axis, self.classifiers_names)
        ax.set_ylabel("Accuracy", fontsize=self.font_size)
        ax.set_ylim([0.5, 1.05])
        ax.tick_params(axis='both', which='major', labelsize=self.label_size)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottom_size),
                  fancybox=True, shadow=True, ncol=3, fontsize=self.font_size)
        # self.__autolabel(bars_literature, ax)
        # self.__autolabel(bars_inf_teory, ax)
        # self.__autolabel(bars_both, ax)
        fig.subplots_adjust(bottom=self.bottom_size)
        plt.savefig(f'{self.__output_folder}/fig1_accuracy__{self.sample_size}.png')

    # Classifier's Training time (ms)
    def plot_trainingtime(self,
                          y_lt_trainingtime, y_lt_std_error_trainingtime,
                          y_inf_trainingtime, y_inf_std_error_trainingtime,
                          y_both_trainingtime, y_both_std_error_trainingtime):
        X_axis = np.arange(len(self.classifiers_names))
        fig, ax = plt.subplots(1, 1, figsize=(17, 9))
        ax.bar(X_axis - self.ax_width, y_lt_trainingtime, self.ax_width,
               label=self.label1,
               yerr=y_lt_std_error_trainingtime, color=self.color1)
        ax.bar(X_axis, y_inf_trainingtime, self.ax_width,
               label=self.label2,
               yerr=y_inf_std_error_trainingtime, color=self.color2)
        ax.bar(X_axis + self.ax_width, y_both_trainingtime, self.ax_width,
               label=self.label3,
               yerr=y_both_std_error_trainingtime, color=self.color3)
        ax.set_xticks(X_axis, self.classifiers_names)
        ax.set_ylabel("Training time (ms)", fontsize=self.font_size)
        ax.tick_params(axis='both', which='major', labelsize=self.label_size)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottom_size),
                  fancybox=True, shadow=True, ncol=3, fontsize=self.font_size)
        fig.subplots_adjust(bottom=self.bottom_size)
        plt.savefig(f'{self.__output_folder}/fig2_training-time__{self.sample_size}.png')

    # Classifier's Data handle time (ms)
    def plot_processingtime(self, time_dataprocessing, time_hc_calculate, ):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        X_axis = np.arange(1)
        ax.bar(X_axis - self.ax_width, [mean(time_dataprocessing) * 1000],
               self.ax_width * 2, yerr=[sem(time_dataprocessing) * 1000],
               label=self.label1, color=self.color1)
        ax.bar(X_axis + self.ax_width, [mean(time_hc_calculate) * 1000],
               self.ax_width * 2, yerr=[sem(time_hc_calculate) * 1000],
               label=self.label2, color=self.color2)
        ax.set_xticks(X_axis, ['Data processing technique'])
        ax.set_ylabel("Data handle time (ms)", fontsize=self.font_size)
        ax.tick_params(axis='both', which='major', labelsize=self.label_size)
        # ax.tick_params(axis='x', labelrotation=45)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottom_size),
                  fancybox=True, shadow=True, ncol=3, fontsize=self.font_size)
        fig.subplots_adjust(bottom=self.bottom_size)
        plt.savefig(f'{self.__output_folder}/fig3_preprocessing-time__{self.sample_size}.png')

    # Classifier's Total time (ms)
    def plot_totaltime(self,
                       y_lt_totaltime, y_lt_std_error_totaltime,
                       y_inf_totaltime, y_inf_std_error_totaltime,
                       y_both_totaltime, y_both_std_error_totaltime):
        X_axis = np.arange(len(self.classifiers_names))
        fig, ax = plt.subplots(1, 1, figsize=(17, 9))
        ax.bar(X_axis - self.ax_width, y_lt_totaltime, self.ax_width,
               label=self.label1,
               yerr=y_lt_std_error_totaltime, color=self.color1)
        ax.bar(X_axis, y_inf_totaltime, self.ax_width,
               label=self.label2,
               yerr=y_inf_std_error_totaltime, color=self.color2)
        ax.bar(X_axis + self.ax_width, y_both_totaltime, self.ax_width,
               label=self.label3,
               yerr=y_both_std_error_totaltime, color=self.color3)
        ax.set_xticks(X_axis, self.classifiers_names)
        ax.set_ylabel("Total time (ms)", fontsize=self.font_size)
        ax.tick_params(axis='both', which='major', labelsize=self.label_size)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottom_size),
                  fancybox=True, shadow=True, ncol=3, fontsize=self.font_size)
        fig.subplots_adjust(bottom=self.bottom_size)
        plt.savefig(f'{self.__output_folder}/fig4_total-time__{self.sample_size}.png')


if __name__ == '__main__':
    classifiers_names = [
        "kNN",
        "Linear SVM",
        "RBF SVM",
        "D. Tree",
        "R. Forest",
        "MLP",
        "N. Bayes",
    ]
    
    samplesize = 9700  # for each dirver

    y_lt_accuracy = [0.9148969072164949, 0.7858762886597939, 0.7893041237113401, 0.8106701030927834, 0.7964690721649486,
                     0.8061443298969072, 0.5060309278350514]
    y_lt_std_error_accuracy = [0.0028815794294779827, 0.008616552895798778, 0.008378592278917449, 0.007976219543154701,
                               0.009246227248882945, 0.0081888089459233, 0.0368496251905744]
    y_inf_accuracy = [0.8125348621010227, 0.7503357091209587, 0.7773163929346143, 0.771005061460593, 0.7617446544778431,
                      0.7753073029645697, 0.7107323623592604]
    y_inf_std_error_accuracy = [0.0032974172808081866, 0.0005400228551886618, 0.004955376179156017,
                                0.0042846860791308896, 0.003235848799990707, 0.004862376736401925, 0.00440546093230214]
    y_both_accuracy = [0.8488585889887409, 0.7851203388079744, 0.8159745894019211, 0.8142392314843507,
                       0.7762111352133044, 0.8078194401404813, 0.5102107220328478]
    y_both_std_error_accuracy = [0.0050174134452188, 0.008969373293902554, 0.00706038490282521, 0.007798272636367851,
                                 0.006441127070440721, 0.0075246133235254845, 0.03643743966983848]

    y_lt_trainingtime = [42.28333234786987, 7051.260590553284, 11189.333045482635, 68.1221604347229, 77.90100574493408,
                         10995.37456035614, 8.212339878082275]
    y_lt_std_error_trainingtime = [2.4394914346650958, 153.0239755181311, 421.0626189880749, 1.2991836399047225,
                                   0.8962166568727636, 1513.8748806443054, 0.0816493389797696]
    y_inf_trainingtime = [43.70490312576294, 7561.359655857086, 16050.565671920776, 115.77626466751099,
                          109.9786639213562, 6312.506866455078, 93.65619421005249]
    y_inf_std_error_trainingtime = [0.4489605505315542, 244.99643839165725, 299.45601930929865, 0.6639426695695417,
                                    0.39065866749728234, 456.9710610584442, 0.4848046832141084]
    y_both_trainingtime = [52.238619327545166, 10344.197833538055, 15182.734489440918, 227.95592546463013,
                           146.580708026886, 11465.88065624237, 111.60821914672852]
    y_both_std_error_trainingtime = [0.33535726007274835, 140.86951113238155, 341.4594751458544, 1.4755611301469367,
                                     0.7820125485557817, 1166.7700729722906, 0.5602884147160432]

    time_dataprocessing = [0.04723954200744629, 0.05038022994995117, 0.04736065864562988, 0.050244808197021484]
    time_hc_calculate = [77.02936935424805, 77.90972566604614, 78.23868584632874, 78.35476756095886, 78.61920475959778,
                         77.892404794693, 78.20450711250305, 78.39171242713928, 78.57746577262878, 80.5811767578125,
                         81.34266686439514, 79.75024652481079, 82.32729053497314, 80.56094026565552, 80.14936184883118,
                         79.83920288085938, 79.78101873397827, 79.18267011642456, 79.37388014793396, 78.72807168960571]

    y_lt_totaltime = [91.08964204788208, 7100.066900253296, 11238.139355182648, 116.92847013473511, 126.70731544494629,
                      11044.180870056152, 57.01864957809448]
    y_lt_std_error_totaltime = [2.4394914346650958, 153.0239755181311, 421.0626189880749, 1.2991836399047225,
                                0.8962166568727636, 1513.8748806443054, 0.0816493389797696]
    y_inf_totaltime = [79285.42338609695, 86803.07813882828, 95292.28415489197, 79357.49474763872, 79351.69714689255,
                       85554.22534942627, 79335.37467718124]
    y_inf_std_error_totaltime = [0.4489605505315542, 244.99643839165725, 299.45601930929865, 0.6639426695695417,
                                 0.39065866749728234, 456.9710610584442, 0.4848046832141084]
    y_both_totaltime = [79342.76341199875, 89634.72262620927, 94473.25928211212, 79518.48071813583, 79437.10550069809,
                        90756.40544891359, 79402.13301181793]
    y_both_std_error_totaltime = [0.33535726007274835, 140.86951113238155, 341.4594751458544, 1.4755611301469367,
                                  0.7820125485557817, 1166.7700729722906, 0.5602884147160432]

    pltr = PlotResults(classifiers_names, samplesize)
    pltr.plot_accuracy(y_lt_accuracy, y_lt_std_error_accuracy,
                       y_inf_accuracy, y_inf_std_error_accuracy,
                       y_both_accuracy, y_both_std_error_accuracy)
    pltr.plot_trainingtime(y_lt_trainingtime, y_lt_std_error_trainingtime,
                           y_inf_trainingtime, y_inf_std_error_trainingtime,
                           y_both_trainingtime, y_both_std_error_trainingtime)
    pltr.plot_processingtime(time_dataprocessing, time_hc_calculate)
    pltr.plot_totaltime(y_lt_totaltime, y_lt_std_error_totaltime,
                        y_inf_totaltime, y_inf_std_error_totaltime,
                        y_both_totaltime, y_both_std_error_totaltime)
    plt.show()
