import numpy as np
from scipy import stats


def diff_calculate(arr1: np.ndarray, arr2: np.ndarray, mask1, mask2):
    sum_arr1 = np.sum(abs(arr1), axis=1)[:, None]
    r_avg1 = np.divide(arr1, sum_arr1, out=np.full_like(arr1, np.nan),
                       where=sum_arr1 != 0)

    sum_arr2 = np.sum(abs(arr2), axis=1)[:, None]
    r_avg2 = np.divide(arr2, sum_arr2, out=np.full_like(arr2, np.nan),
                       where=sum_arr2 != 0)
    avg1 = np.average(abs(arr1), axis=0)
    avg2 = np.average(abs(arr2), axis=0)

    mask = (mask1 * mask2).astype(bool)

    order1 = np.argsort(avg1 * mask1)[::-1]
    order2 = np.argsort(avg2 * mask2)[::-1]
    for i, m in enumerate(mask1):
        if not m:
            order1 = np.delete(order1, np.argwhere(order1 == i))
            order1 = np.append(order1, i)
    for i, m in enumerate(mask2):
        if not m:
            order2 = np.delete(order2, np.argwhere(order2 == i))
            order2 = np.append(order2, i)

    diff_inf = [np.average(abs(arr1 - arr2), axis=0)[i] for i, m in enumerate(mask) if m]
    diff_rinf = [np.average(abs(r_avg1 - r_avg2), axis=0)[i] for i, m in enumerate(mask) if m]
    diff_rank = [abs(order1.tolist().index(i) - order2.tolist().index(i)) / len(mask) for i, m in enumerate(mask) if m]
    return diff_inf, diff_rinf, diff_rank


def tau_kendall(arr1: np.ndarray, arr2: np.ndarray, mask1, mask2, acc_ref, acc_2) -> float:
    avg1 = np.average(abs(arr1), axis=0)
    avg2 = np.average(abs(arr2), axis=0)

    mask = (mask1 * mask2).astype(bool)
    avg1 = np.argsort(avg1*mask)[::-1][:sum(mask)]
    avg2 = np.argsort(avg2*mask)[::-1][:sum(mask)]

    tau, p_value = stats.kendalltau(avg1, avg2)
    return tau


def relative_influence_changes(arr1: np.ndarray, arr2: np.ndarray,
                                  mask1, mask2, acc_ref, acc_2) -> float:
    sum_arr1 = np.sum(abs(arr1), axis=1)[:, None]
    r_avg1 = np.average(np.divide(arr1, sum_arr1, out=np.full_like(arr1, np.nan),
                                  where=sum_arr1 != 0), axis=0)

    sum_arr2 = np.sum(abs(arr2), axis=1)[:, None]
    r_avg2 = np.average(np.divide(arr2, sum_arr2, out=np.full_like(arr2, np.nan),
                                  where=sum_arr2 != 0), axis=0)

    mask = (mask1 * mask2).astype(bool)
    return np.sum(abs(r_avg1-r_avg2))


def RI(arr_ref: np.ndarray, arr2: np.ndarray, mask_ref, mask2, acc_ref, acc_2):
    esp = np.finfo(float).eps
    _, diff_rinf, diff_rank = diff_calculate(arr_ref, arr2, mask_ref, mask2)

    diff_rinf = np.add(np.power(diff_rinf, 1 / 4), esp)
    diff_rank = np.add(diff_rank, esp)
    return np.average(np.multiply(diff_rinf, diff_rank) - esp ** 2)


def RIA(arr_ref: np.ndarray, arr2: np.ndarray, mask_ref, mask2, acc_ref, acc_2):
    esp = np.finfo(float).eps
    _, diff_rinf, diff_rank = diff_calculate(arr_ref, arr2, mask_ref, mask2)

    diff_rinf = np.add(np.power(diff_rinf, 1 / 4), esp)
    diff_rank = np.add(diff_rank, esp)
    diff_acc = acc_2 - acc_ref + esp
    return np.average(np.multiply(diff_rinf, diff_rank) * diff_acc - esp ** 3)