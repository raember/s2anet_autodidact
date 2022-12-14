import argparse
import json
import os
import pickle
from pathlib import Path

import mmcv
import numpy as np
import pandas as pd
from obb_anns import OBBAnns


def parse_args():
    parser = argparse.ArgumentParser(description='Compare Overlap between Models')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('jsons_gt', nargs='+', default=[],
                        help='Path to JSON file(s) with proposals which is used as ground truth')
    parser.add_argument('--jsons_pr', nargs='+', default=[], help='Path to JSON file(s) with proposals')
    parser.add_argument('--out', default=None, help='Path where to save metrics')
    args = parser.parse_args()
    return args


def _calc_area(corners):
    area = 0.0
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def _prop_to_ann(proposals, obb):
    annotations = {}

    for i, prop in enumerate(proposals['proposals']):
        annotations[i] = {
            'a_bbox': [0, 0, 0, 0],  # does not matter since only oriented boxes are used
            'o_bbox': [float(x) for x in prop['bbox']],
            'cat_id': [int(prop['cat_id'])],
            'area': _calc_area(
                [(float(prop['bbox'][i]), float(prop['bbox'][i + 1])) for i in range(len(prop['bbox']) - 1)]),
            'img_id': obb.img_idx_lookup[prop['img_id']],
            'comments': "",
        }

    return annotations


def _proposal_to_df(obb, json_fp):
    f = open(json_fp)
    props = json.load(f)
    f.close()

    df = pd.DataFrame.from_dict(_prop_to_ann(props, obb)).transpose()
    return df


def _get_anns_custom(df):
    def get_anns_f(img_idx):
        return df[df.img_id == img_idx]

    return get_anns_f


def _get_custom_class_gt(df):
    def count_class_gt_f(class_id):
        if class_id is not None:
            ann_gt_idxs = set(df[df['cat_id'].map(lambda x: int(x[0])) == class_id].index)
        else:
            ann_gt_idxs = set(df.index)

        return ann_gt_idxs

    return count_class_gt_f


def _calculate_metrics(o, _get_anns_f, count_class_gt_f, proposal_fp):
    o.load_proposals(proposal_fp)
    o.get_anns = _get_anns_f
    o._count_class_gt = count_class_gt_f

    metric_results = o.calculate_metrics(iou_thrs=np.array([0.5, 0.8, 0.9]), classwise=False, average_thrs=False)
    categories = o.get_cats()
    occurences_by_class = o.get_class_occurences()

    return metric_results, categories, occurences_by_class


def _store_results_classwise(work_dir, filename, metric_results, categories, occurences_by_class):
    metric_results = {categories[key]['name']: value for (key, value) in metric_results.items()}

    for (key, value) in metric_results.items():
        value.update(no_occurences=occurences_by_class[key])

    if work_dir is not None and work_dir != "":
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
        out_file = os.path.join(work_dir, filename)
        pickle.dump(metric_results, open(out_file, 'wb'))

    print(metric_results)


def _store_results(filename, metric_results):
    pickle.dump(metric_results, open(filename, 'wb'))
    print(metric_results)


def _read_results(filename):
    print(f"Read existing results from {filename}...")
    metric_results = pickle.load(open(filename, 'rb'))
    return metric_results


def _setup_obb_anns(cfg):
    o = OBBAnns(cfg.data.test.ann_file)
    o.load_annotations()
    o.set_annotation_set_filter(['deepscores'])
    o.props_oriented = True

    return o


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    base_path = Path(args.out)
    if not base_path.exists():
        base_path.mkdir()

    obb = _setup_obb_anns(cfg)

    rows = {}
    columns = []
    filename = "overlap.pkl"

    for json_gt in args.jsons_gt:
        df = _proposal_to_df(obb, json_gt)
        get_anns_f = _get_anns_custom(df)
        count_class_gt_f = _get_custom_class_gt(df)
        for json_pr in args.jsons_pr:
            if json_gt != json_pr:

                f1 = json_gt.split("/")[-3] if "/" in json_gt else json_gt
                f2 = json_pr.split("/")[-3] if "/" in json_pr else json_pr

                out_path = base_path / f"{f1}-{f2}/"
                if not out_path.exists():
                    out_path.mkdir()
                out_fp = out_path / filename

                if out_fp.exists():
                    metric_results = _read_results(out_fp)
                else:
                    metric_results, categories, occurences_by_class = _calculate_metrics(obb, get_anns_f,
                                                                                         count_class_gt_f,
                                                                                         json_pr)

                    _store_results(out_fp, metric_results)

                if json_pr in rows:
                    rows[json_pr].append(metric_results['average'][0.5]['ap'])
                else:
                    rows[json_pr] = [metric_results['average'][0.5]['ap']]

            else:
                if json_pr in rows:
                    rows[json_pr].append(1.)
                else:
                    rows[json_pr] = [1.]

        columns.append(json_gt)

    pd.DataFrame.from_dict(rows, orient='index', columns=columns).to_csv(base_path / "overlap_matrix.csv")


if __name__ == '__main__':
    main()
