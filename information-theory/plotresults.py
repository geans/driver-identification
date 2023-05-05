import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from scipy.stats import sem

classifiers_names = [
    "kNN",
    "Linear SVM",
    "RBF SVM",
    "D. Tree",
    "R. Forest",
    "MLP",
    "N. Bayes",
]


# samplesize = 300 # for each dirver

# y_lt_accuracy             = [0.9766666666666668, 0.7599999999999999, 0.95, 0.9445, 0.9288333333333334, 0.9591666666666665, 0.7825]
# y_lt_std_error_accuracy   = [0.004223299339372181, 0.0030107021581983356, 0.007628049193792862, 0.010725918218848536, 0.010663033920871713, 0.006608850171156501, 0.02902093920397308]
# y_inf_accuracy            = [0.8734875444839858, 0.7480427046263344, 0.8517793594306049, 0.8026690391459075, 0.7838078291814947, 0.8419928825622776, 0.5661921708185054]
# y_inf_std_error_accuracy  = [0.006569440473131856, 0.005552897971266728, 0.009993385620871192, 0.007824391083280603, 0.00853887641655082, 0.010930357010624602, 0.012365284707091197]
# y_both_accuracy           = [0.9459074733096084, 0.7494661921708186, 0.9868327402135231, 0.9494661921708184, 0.8590747330960852, 0.9944839857651246, 0.7882562277580071]
# y_both_std_error_accuracy = [0.007213252801469515, 0.005172560317451981, 0.0024911032028469777, 0.009296508875378508, 0.011902473349748028, 0.0022280356500738153, 0.02735392971762329]

# y_lt_trainingtime             = [0.9773612022399903, 6.459379196166992, 5.366301536560059, 1.8718957901000977, 8.523499965667725, 727.9577732086182, 0.8417844772338867]
# y_lt_std_error_trainingtime   = [0.04147497419918877, 0.08555939215877367, 0.27517656244414535, 0.04241016549495803, 0.08052245213459262, 35.08579699404084, 0.01980877979348139]
# y_inf_trainingtime            = [0.911867618560791, 6.261897087097168, 9.170842170715332, 3.032374382019043, 8.793067932128906, 804.6461820602417, 1.1357545852661133]
# y_inf_std_error_trainingtime  = [0.008387193202566296, 0.09157476501917201, 0.20821925377070447, 0.07374108749373633, 0.1412165293387949, 41.751733171989564, 0.011149225008152084]
# y_both_trainingtime           = [0.9975910186767578, 6.2589287757873535, 7.066977024078369, 3.6914467811584473, 9.265518188476562, 597.3753929138184, 1.2368559837341309]
# y_both_std_error_trainingtime = [0.009012952587481058, 0.05918390771975289, 0.28731990641466576, 0.2013243277163334, 0.15826362219338258, 34.32367084832547, 0.009244498468032897]

# time_dataprocessing = [0.009368896484375, 0.00879216194152832, 0.009892702102661133, 0.009431600570678711]
# time_hc_calculate   = [2.1431705951690674, 2.1211578845977783, 2.116119146347046, 2.1357171535491943, 2.1352641582489014, 2.4400129318237305, 2.2199978828430176, 2.2542357444763184, 2.206141948699951, 2.2078161239624023, 2.2138381004333496, 2.201542854309082, 2.2094314098358154, 2.218409538269043, 2.218695878982544, 2.2231576442718506, 2.203444004058838, 2.1920883655548096, 2.1952028274536133, 2.202549934387207]

# y_lt_totaltime             = [10.348701477050781, 15.830719470977785, 14.73764181137085, 11.243236064910889, 17.894840240478516, 737.329113483429, 10.213124752044678]
# y_lt_std_error_totaltime   = [0.04147497419918877, 0.08555939215877367, 0.27517656244414535, 0.04241016549495803, 0.08052245213459262, 35.08579699404084, 0.01980877979348139]
# y_inf_totaltime            = [2203.811573982239, 2209.1616034507756, 2212.0705485343938, 2205.932080745697, 2211.6927742958073, 3007.5458884239197, 2204.035460948944]
# y_inf_std_error_totaltime  = [0.008387193202566296, 0.09157476501917201, 0.20821925377070447, 0.07374108749373633, 0.1412165293387949, 41.751733171989564, 0.011149225008152084]
# y_both_totaltime           = [2213.2686376571655, 2218.529975414276, 2219.3380236625676, 2215.9624934196477, 2221.536564826966, 2809.646439552307, 2213.507902622223]
# y_both_std_error_totaltime = [0.009012952587481058, 0.05918390771975289, 0.28731990641466576, 0.2013243277163334, 0.15826362219338258, 34.32367084832547, 0.009244498468032897]


samplesize = 9700 # for each dirver

y_lt_accuracy             = [0.9148969072164949, 0.7858762886597939, 0.7893041237113401, 0.8106701030927834, 0.7964690721649486, 0.8061443298969072, 0.5060309278350514]
y_lt_std_error_accuracy   = [0.0028815794294779827, 0.008616552895798778, 0.008378592278917449, 0.007976219543154701, 0.009246227248882945, 0.0081888089459233, 0.0368496251905744]
y_inf_accuracy            = [0.8125348621010227, 0.7503357091209587, 0.7773163929346143, 0.771005061460593, 0.7617446544778431, 0.7753073029645697, 0.7107323623592604]
y_inf_std_error_accuracy  = [0.0032974172808081866, 0.0005400228551886618, 0.004955376179156017, 0.0042846860791308896, 0.003235848799990707, 0.004862376736401925, 0.00440546093230214]
y_both_accuracy           = [0.8488585889887409, 0.7851203388079744, 0.8159745894019211, 0.8142392314843507, 0.7762111352133044, 0.8078194401404813, 0.5102107220328478]
y_both_std_error_accuracy = [0.0050174134452188, 0.008969373293902554, 0.00706038490282521, 0.007798272636367851, 0.006441127070440721, 0.0075246133235254845, 0.03643743966983848]

y_lt_trainingtime             = [42.28333234786987, 7051.260590553284, 11189.333045482635, 68.1221604347229, 77.90100574493408, 10995.37456035614, 8.212339878082275]
y_lt_std_error_trainingtime   = [2.4394914346650958, 153.0239755181311, 421.0626189880749, 1.2991836399047225, 0.8962166568727636, 1513.8748806443054, 0.0816493389797696]
y_inf_trainingtime            = [43.70490312576294, 7561.359655857086, 16050.565671920776, 115.77626466751099, 109.9786639213562, 6312.506866455078, 93.65619421005249]
y_inf_std_error_trainingtime  = [0.4489605505315542, 244.99643839165725, 299.45601930929865, 0.6639426695695417, 0.39065866749728234, 456.9710610584442, 0.4848046832141084]
y_both_trainingtime           = [52.238619327545166, 10344.197833538055, 15182.734489440918, 227.95592546463013, 146.580708026886, 11465.88065624237, 111.60821914672852]
y_both_std_error_trainingtime = [0.33535726007274835, 140.86951113238155, 341.4594751458544, 1.4755611301469367, 0.7820125485557817, 1166.7700729722906, 0.5602884147160432]

time_dataprocessing = [0.04723954200744629, 0.05038022994995117, 0.04736065864562988, 0.050244808197021484]
time_hc_calculate   = [77.02936935424805, 77.90972566604614, 78.23868584632874, 78.35476756095886, 78.61920475959778, 77.892404794693, 78.20450711250305, 78.39171242713928, 78.57746577262878, 80.5811767578125, 81.34266686439514, 79.75024652481079, 82.32729053497314, 80.56094026565552, 80.14936184883118, 79.83920288085938, 79.78101873397827, 79.18267011642456, 79.37388014793396, 78.72807168960571]

y_lt_totaltime             = [91.08964204788208, 7100.066900253296, 11238.139355182648, 116.92847013473511, 126.70731544494629, 11044.180870056152, 57.01864957809448]
y_lt_std_error_totaltime   = [2.4394914346650958, 153.0239755181311, 421.0626189880749, 1.2991836399047225, 0.8962166568727636, 1513.8748806443054, 0.0816493389797696]
y_inf_totaltime            = [79285.42338609695, 86803.07813882828, 95292.28415489197, 79357.49474763872, 79351.69714689255, 85554.22534942627, 79335.37467718124]
y_inf_std_error_totaltime  = [0.4489605505315542, 244.99643839165725, 299.45601930929865, 0.6639426695695417, 0.39065866749728234, 456.9710610584442, 0.4848046832141084]
y_both_totaltime           = [79342.76341199875, 89634.72262620927, 94473.25928211212, 79518.48071813583, 79437.10550069809, 90756.40544891359, 79402.13301181793]
y_both_std_error_totaltime = [0.33535726007274835, 140.86951113238155, 341.4594751458544, 1.4755611301469367, 0.7820125485557817, 1166.7700729722906, 0.5602884147160432]


class PlotResults:
  def __init__(self,
                classifiers_names,
                samplesize,
                labelsize = 26,
                fontsize = 24,
                ax_width = 0.3,
                bottomsize = 0.3):
    self.classifiers_names = classifiers_names
    self.samplesize = samplesize
    self.labelsize = labelsize
    self.fontsize = fontsize
    self.ax_width = ax_width
    self.bottomsize = bottomsize

  def __autolabel(self, rects, ax, vertical_pos=.5):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., vertical_pos,
                f'{height:3.2}', fontsize=15,
                ha='center', va='bottom')

  def plot_accuracy(self,
                    y_lt_accuracy, y_lt_std_error_accuracy,
                    y_inf_accuracy, y_inf_std_error_accuracy,
                    y_both_accuracy, y_both_std_error_accuracy):
    X_axis = np.arange(len(self.classifiers_names))
    fig, ax = plt.subplots(1, 1, figsize=(16,9))

    # Classifier's Accuracy
    bars_literature = ax.bar(X_axis - self.ax_width, y_lt_accuracy, self.ax_width,
                        label='Literature', 
                        yerr=y_lt_std_error_accuracy, color='deepskyblue')
    bars_inf_teory  = ax.bar(X_axis, y_inf_accuracy, self.ax_width,
                        label='Complexity-Entropy', 
                        yerr=y_inf_std_error_accuracy, color='orange')
    bars_both       = ax.bar(X_axis + self.ax_width, y_both_accuracy, self.ax_width,
                        label='Literature + Complexity-Entropy', 
                        yerr=y_both_std_error_accuracy, color='lime')
    ax.set_xticks(X_axis, self.classifiers_names)
    ax.set_ylabel("Accuracy", fontsize=self.fontsize)
    ax.set_ylim([0.5, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottomsize),
          fancybox=True, shadow=True, ncol=3, fontsize=self.fontsize)
    self.__autolabel(bars_literature, ax)
    self.__autolabel(bars_inf_teory, ax)
    self.__autolabel(bars_both, ax)
    fig.subplots_adjust(bottom=self.bottomsize)
    plt.savefig(f'fig1_accuracy__{self.samplesize}.png')

  # plt.show()
  # exit()

  # Classifier's Training time (ms)
  def plot_trainingtime(self,
                    y_lt_trainingtime, y_lt_std_error_trainingtime,
                    y_inf_trainingtime, y_inf_std_error_trainingtime,
                    y_both_trainingtime, y_both_std_error_trainingtime):
    X_axis = np.arange(len(self.classifiers_names))
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    bars_literature = ax.bar(X_axis - self.ax_width, y_lt_trainingtime, self.ax_width,
                        label = 'Literature',
                        yerr=y_lt_std_error_trainingtime, color='deepskyblue')
    bars_inf_teory  = ax.bar(X_axis, y_inf_trainingtime, self.ax_width,
                        label = 'Complexity-Entropy',
                        yerr=y_inf_std_error_trainingtime, color='orange')
    bars_both       = ax.bar(X_axis + self.ax_width, y_both_trainingtime, self.ax_width,
                        label='Literature + Complexity-Entropy', 
                        yerr=y_both_std_error_trainingtime, color='lime')
    ax.set_xticks(X_axis, self.classifiers_names)
    ax.set_ylabel("Training time (ms)", fontsize=self.fontsize)
    ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottomsize),
          fancybox=True, shadow=True, ncol=3, fontsize=self.fontsize)
    fig.subplots_adjust(bottom=self.bottomsize)
    plt.savefig(f'fig2_training-time__{self.samplesize}.png')

  # Classifier's Data handle time (ms)
  def plot_processingtime(self, time_dataprocessing, time_hc_calculate,):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    X_axis = np.arange(1)
    bars_literature = ax.bar(X_axis - self.ax_width, [mean(time_dataprocessing) * 1000],
                        self.ax_width * 2, yerr=[sem(time_dataprocessing) * 1000],
                        label = 'Literature', color='deepskyblue')
    bars_inf_teory  = ax.bar(X_axis + self.ax_width, [mean(time_hc_calculate) * 1000],
                        self.ax_width * 2, yerr=[sem(time_hc_calculate) * 1000],
                        label = 'Complexity-Entropy', color='orange')
    ax.set_xticks(X_axis, ['Data processing technique'])
    ax.set_ylabel("Data handle time (ms)", fontsize=self.fontsize)
    ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
    # ax.tick_params(axis='x', labelrotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottomsize),
          fancybox=True, shadow=True, ncol=3, fontsize=self.fontsize)
    fig.subplots_adjust(bottom=self.bottomsize)
    plt.savefig(f'fig3_preprocessing-time__{self.samplesize}.png')

  # Classifier's Total time (ms)
  def plot_totaltime(self,
                    y_lt_totaltime, y_lt_std_error_totaltime,
                    y_inf_totaltime, y_inf_std_error_totaltime,
                    y_both_totaltime, y_both_std_error_totaltime):
    X_axis = np.arange(len(self.classifiers_names))
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    bars_literature = ax.bar(X_axis - self.ax_width, y_lt_totaltime, self.ax_width,
                        label = 'Literature',
                        yerr=y_lt_std_error_totaltime, color='deepskyblue')
    bars_inf_teory  = ax.bar(X_axis, y_inf_totaltime, self.ax_width,
                        label = 'Complexity-Entropy',
                        yerr=y_inf_std_error_totaltime, color='orange')
    bars_both       = ax.bar(X_axis + self.ax_width, y_both_totaltime, self.ax_width,
                        label='Literature + Complexity-Entropy', 
                        yerr=y_both_std_error_totaltime, color='lime')
    ax.set_xticks(X_axis, self.classifiers_names)
    ax.set_ylabel("Total time (ms)", fontsize=self.fontsize)
    ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -self.bottomsize),
          fancybox=True, shadow=True, ncol=3, fontsize=self.fontsize)
    fig.subplots_adjust(bottom=self.bottomsize)
    plt.savefig(f'fig4_total-time__{self.samplesize}.png')


if __name__ == '__main__':
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