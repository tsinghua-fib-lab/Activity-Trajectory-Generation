import numpy as np
import scipy.stats
import pandas as pd
import math
from bisect import bisect_right
import utils

class Evaluation(object):

    def __init__(self, args):
        self.args= args

    def arr_to_distribution(self,arr, Min, Max, bins, over=None):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """

        distribution, base = np.histogram(arr[arr<=Max],bins = bins,range=(Min,Max))
        m = np.array([len(arr[arr>Max])],dtype='int64')
        distribution = np.hstack((distribution,m))


        return distribution, base[:-1]

    def get_js_divergence(self, p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-9)
        p2 = p2 / (p2.sum()+1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def hour(self,p1,p2):
        
        f = [int((i[0]%1)*24) for u in p1 for i in u]
        r = [int((i[0]%1)*24) for u in p2 for i in u]

        f = pd.value_counts(f,normalize=True)
        r = pd.value_counts(r,normalize=True)

        f_list = [(f.keys()[i],f.values[i]) for i in len(f)]
        r_list = [(r.keys()[i],r.values[i]) for i in len(r)]

        for i in range(24):
            if i not in f.keys():
                f_list.append((0,0.0))
            if i not in r.keys():
                r_list.append((0,0.0))

        f_list.sort()
        r_list.sort()

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def week(self,p1,p2):
        if self.args.data=='Mobile':
            f = [[1.0 if i[0]%7>=5 else 0.0 for i in u] for u in p1]
            r = [[1.0 if i[0]%7>=5 else 0.0 for i in u] for u in p2]
            f = [sum(u)/(len(u)+1e-9) for u in f]
            r = [sum(u)/(len(u)+1e-9) for u in r]
        
        MIN = 0
        MAX = 1.0

        bins = 10

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def distance_one_step(self, p1, p2):
        f = [utils.geodistance(i[2][0],i[2][1],u[index][2][0],u[index][2][1]) for u in p1 for index, i in enumerate(u[1:])]
        r = [utils.geodistance(i[2][0],i[2][1],u[index][2][0],u[index][2][1]) for u in p2 for index, i in enumerate(u[1:])]

        MIN = 0
        MAX = 10
        bins = 10

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        r = r_list / (r_list.sum()+1e-9)
        f = f_list / (f_list.sum()+1e-9)



        JSD = self.get_js_divergence(r_list, f_list)

        return JSD


    def distance_jsd(self,p1,p2):

        f = [sum([utils.geodistance(i[2][0],i[2][1],u[index][2][0],u[index][2][1]) for index, i in enumerate(u[1:])]) for u in p1]
        r = [sum([utils.geodistance(i[2][0],i[2][1],u[index][2][0],u[index][2][1]) for index, i in enumerate(u[1:])]) for u in p2]

        
        MIN = 0
        MAX = 500
        bins = 10

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        r = r_list / (r_list.sum()+1e-9)
        f = f_list / (f_list.sum()+1e-9)


        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def radius_jsd(self, p1, p2):

        def radius_cal(p):
            c = np.mean(p,axis=0)
            r = np.mean([utils.geodistance(i[0],i[1],c[0],c[1]) for i in p])
            return r

        f = [round(radius_cal(np.array([(i[2][0],i[2][1]) for i in u]))) for u in p1]
        r = [round(radius_cal(np.array([(i[2][0],i[2][1]) for i in u]))) for u in p2]

        MIN = 0
        MAX = 30
        bins = 6

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        r = r_list / (r_list.sum()+1e-9)
        f = f_list / (f_list.sum()+1e-9)

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def duration_jsd(self,p1,p2):

        def duration(p):
            d = [[i[0]-u[index][0] for index, i in enumerate(u[1:])] for u in p]
            d = [round(i*10) for u in d for i in u]
            return d

        f = duration(p1)
        r = duration(p2)

        MIN = 0
        MAX = 15
        bins = math.ceil(MAX-MIN)


        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def dailyloc_jsd(self, p1, p2):

        f = [round(len(u)/31) for u in p1]

        r = [round(len(u)/31) for u in p2]

        MIN = 0
        MAX = 4
        bins = math.ceil(MAX-MIN)

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def need_jsd(self, p1, p2):

        rt = pd.value_counts([i[1] for u in p2 for i in u],normalize=True)
        r = [(rt.keys()[i],rt.values[i]) for i in range(len(rt))]
        r.sort()
        r_list = [i[1] for i in r]

        ft = pd.value_counts([i[1] for u in p1 for i in u],normalize=True)
        f = [(ft.keys()[i],ft.values[i]) for i in range(len(ft))]
        for i in range(self.args.num_event):
            if i not in ft.keys():
                f.append((i,0))
        f.sort()
        f_list = [i[1] for i in f]

        r_array = np.array(r_list)
        f_array = np.array(f_list)

        JSD = self.get_js_divergence(r_array, f_array)

        return JSD

    def get_JSD(self,real,fake):

        distance_jsd = self.distance_jsd(real,fake)
        radius_jsd = self.radius_jsd(real,fake)
        duration_jsd = self.duration_jsd(real,fake)
        dailyloc_jsd = self.dailyloc_jsd(real,fake)
        distance_step = self.distance_one_step(real, fake)

        return distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd, distance_step