"""
Load the results from the paper's results and compute the final stat to report.
"""

import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="path to evaluation csv")
args = parser.parse_args()


# parse the csv
# columns are sample_time,epoch,degree_validate,clustering_validate,orbits4_validate,degree_test,clustering_test,orbits4_test
with open(args.csv, "r") as f:
    reader = csv.DictReader(
        f,
    )

    results = []

    for row in reader:
        results.append(
            (
                int(row["epoch"]),
                dict(
                    degree_validate=float(row["degree_validate"]),
                    clustering_validate=float(row["clustering_validate"]),
                    orbits4_validate=float(row["orbits4_validate"]),
                    degree_test=float(row["degree_test"]),
                    clustering_test=float(row["clustering_test"]),
                    orbits4_test=float(row["orbits4_test"]),
                ),
            )
        )

    # compute early stopping
    # do early stopping on degree mmd
    min_deg_mmd, min_deg_epoch = float("inf"), 0
    for epoch, res in results:
        if res["degree_validate"] < min_deg_mmd:
            min_deg_mmd = res["degree_validate"]
            min_deg_epoch = epoch
        elif epoch - min_deg_epoch > 300:
            print("Early stopping at epoch {}".format(epoch))
            break

    # find the result for the epoch with the lowest degree mmd
    es_result = [r for e, r in results if e == min_deg_epoch][0]
    print("ES epoch: {}, results: {}".format(min_deg_epoch, es_result))

    def argminimum(key):
        items = [(k, v[key]) for k, v in results]
        e, v = min(items, key=lambda x: x[1])
        return e, v

    print()
    print(
        "Best results overall: degree={}, clustering={}, orbits={}".format(
            argminimum("degree_test"),
            argminimum("clustering_test"),
            argminimum("orbits4_test"),
        )
    )
