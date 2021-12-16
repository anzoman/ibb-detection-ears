import glob
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plot
import numpy as np

from metrics.evaluation import Evaluation
from preprocessing.preprocess import Preprocess


class EvaluateAll:
    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_evaluation(self, detector, iou_threshold, print_results=False):
        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        y_true = []
        preprocess = Preprocess()
        evaluation = Evaluation()
        t = time.process_time()

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            # img = preprocess.histogram_equalization_rgb(img)  # This one makes VJ worse

            # Run the detector. It runs a list of all the detected bounding-boxes.
            # In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = detector.detect(img)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            # Only for detection:
            p, gt = evaluation.prepare_for_detection(prediction_list, annot_list)

            iou = evaluation.iou_compute(p, gt)
            iou_arr.append(iou)

            if len(annot_list) > 0:
                y_true.append("positive")
            else:
                y_true.append("negative")

            if len(annot_list) > 0 and len(prediction_list) == 0:
                fn += 1
            elif iou >= iou_threshold:
                tp += 1
            elif iou < iou_threshold:
                fp += 1
            else:
                tn += 1

        elapsed_time = time.process_time() - t
        miou = np.average(np.array(iou_arr))
        accuracy = evaluation.accuracy(tp, fp, fn, tn)
        precision = evaluation.precision(tp, fp, fn)
        recall = evaluation.recall(tp, fn)
        f1 = evaluation.f1_score(precision, recall)
        thresholds = np.arange(start=0.2, stop=0.9, step=0.05)
        precisions, recalls = evaluation.precision_recall_curve(y_true, iou_arr, thresholds)
        ap = evaluation.ap(np.array(precisions), np.array(recalls))

        if print_results:
            print()
            print("Average IOU:", f"{miou:.2%}")
            print("TP:", tp)
            print("FP:", fp)
            print("FN:", fn)
            print("Accuracy:", f"{accuracy:.2%}")
            print("Precision:", f"{precision:.2%}")
            print("Recall:", f"{recall:.2%}")
            print("F1 score:", f"{f1:.2%}")
            print("(Mean) Average Precision:", f"{ap:.2%}")
            print("Elapsed time:", f"{elapsed_time}s")

        return miou, tp, fp, fn, accuracy, precision, recall, f1, ap, elapsed_time

    def evaluate_cascade_detector(self, detector, scale_factor, min_neighbors, iou_threshold, print_results=False):
        detector.set_scale_factor(scale_factor)
        detector.set_min_neighbours(min_neighbors)
        return self.run_evaluation(detector, iou_threshold, print_results)

    def evaluate_cascade_detector_with_params(self, detector, scale_factors_list, min_neighbors_list, iou_threshold,
                                              print_results=False):
        results = []
        params = [(s, m) for s in scale_factors_list for m in min_neighbors_list]
        for scale_factor, min_neighbors in params:
            result = self.evaluate_cascade_detector(detector, scale_factor, min_neighbors, iou_threshold, print_results)
            results.append((scale_factor, min_neighbors) + result)
        return results

    def evaluate_cascade_detector_find_best_result(self, detector, scale_factors_list, min_neighbors_list,
                                                   iou_threshold):
        cascade_results = self.evaluate_cascade_detector_with_params(detector, scale_factors_list, min_neighbors_list,
                                                                     iou_threshold)

        best_iou = 0
        best_result = None
        for r in cascade_results:
            iou = r[2]
            if iou > best_iou:
                best_iou = iou
                best_result = r

        return best_result

    def plot_graph(self, x, y, title, x_label, y_label, save_path=None):
        plot.xlabel(x_label)
        plot.ylabel(y_label)
        plot.title(title)
        plot.grid()
        plot.plot(x, y)

        if save_path:
            plot.savefig(save_path)

        plot.show()
        plot.clf()


if __name__ == '__main__':
    # import detectors
    import detectors.cascade_detector.cascade_detector_ears1 as cascade_detector1
    import detectors.cascade_detector.cascade_detector_ears2 as cascade_detector2
    import detectors.cascade_detector.cascade_detector_ears3 as cascade_detector3
    import detectors.cascade_detector.cascade_detector_ears4 as cascade_detector4
    import detectors.yolo_detector.yolov5_detector_ears_1 as yolo_detector1
    import detectors.yolo_detector.yolov5_detector_ears_2 as yolo_detector2
    import detectors.yolo_detector.yolov5_detector_ears_3 as yolo_detector3
    import detectors.yolo_detector.yolov5_detector_ears_4 as yolo_detector4
    import detectors.yolo_detector.yolov5_detector_ears_5 as yolo_detector5

    # create detector objects
    cascade_detector_ears1 = cascade_detector1.CascadeDetectorEars1()
    cascade_detector_ears2 = cascade_detector2.CascadeDetectorEars2()
    cascade_detector_ears3 = cascade_detector3.CascadeDetectorEars3()
    cascade_detector_ears4 = cascade_detector4.CascadeDetectorEars4()
    yolo_detector_ears1 = yolo_detector1.YOLOv5DetectorEars1()
    yolo_detector_ears2 = yolo_detector2.YOLOv5DetectorEars2()
    yolo_detector_ears3 = yolo_detector3.YOLOv5DetectorEars3()
    yolo_detector_ears4 = yolo_detector4.YOLOv5DetectorEars4()
    yolo_detector_ears5 = yolo_detector5.YOLOv5DetectorEars5()

    # create evaluation object
    ev = EvaluateAll()

    # This will evaluate just one VJ cascade detector
    ev.evaluate_cascade_detector(cascade_detector_ears1, 1.01, 2, 0.5, True)

    # This will evaluate just one cascade detector with multiple combinations of detection params
    # scale_factor_arr = np.arange(start=1.05, stop=1.5, step=0.05)
    # min_neighbors_arr = np.arange(start=1, stop=12, step=1)
    # ev.evaluate_cascade_detector_with_params(cascade_detector_ears1, scale_factor_arr, min_neighbors_arr, 0.5)

    # This function finds the best VJ result for given parameters
    # ev.evaluate_cascade_detector_find_best_result(cascade_detector_ears1, scale_factor_arr, min_neighbors_arr, 0.5)

    # This will create two plots for VJ with average IoU based on scale factor and min neighbours
    # scale_factor_plot = []
    # neighbours_plot = []
    # scale_factors = [1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # num_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #
    # for scale_factor in scale_factors:
    #     print(scale_factor)
    #     scale_factor_plot.append(ev.evaluate_cascade_detector(cascade_detector_ears1, scale_factor, 2, 0.5)[0])
    #
    # ev.plot_graph(scale_factors, scale_factor_plot, 'Effect of scale factor on IoU', 'Scale factor', 'Average IoU',
    #               save_path='vj_scale_factor_plot.png')
    #
    # for n in num_neighbors:
    #     print(n)
    #     neighbours_plot.append(ev.evaluate_cascade_detector(cascade_detector_ears1, 1.01, n, 0.5)[0])
    #
    # ev.plot_graph(num_neighbors, neighbours_plot, 'Effect of minimal number of neighbours on IoU',
    #               'Minimal number of neighbours', 'Average IoU', save_path='vj_min_neighbours_plot.png')

    # This will evaluate just one YOLOv5 cascade detector
    ev.run_evaluation(yolo_detector_ears1, 0.5, True)
