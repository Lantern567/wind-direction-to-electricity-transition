"""
§2.3 双层机制重算：尾流链形成 -> 功率曲线转化 -> 年电量差
================================================================

目标
----
把“方向响应幅度”拆成可审计的中间物理量，而不是继续用一个综合指标解释另一个综合指标。

尾流层（固定 9 m/s）
  1. 有效上游尾流数：单个上游机组在轮毂处造成 >=2% 速度亏损；
  2. 多重尾流机组比例：同时受到 >=2 条有效尾流影响；
  3. 串联尾流链长度：有效尾流有向图中的最长路径；
  4. 平均/P90 速度亏损、重度功率损失机组比例、场级效率。

发电层（3--25 m/s）
  1. 在同一真实排布的最不利与最有利来流方向之间逐风速计算功率恢复；
  2. 用各场 Weibull 风速分布加权，分解 3--6、7--10、11--14、15--25 m/s
     对年电量恢复的贡献；
  3. 给出尾流速度恢复经功率曲线转化后的年能量增量。

说明
----
这里用带转子重叠面积和平方和叠加的解析 Jensen 模型做机制诊断。
它不是对 FLORIS + ERA5 逐时回测的替代，而是用独立、可解释的中间量打开物理过程。
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

COORD = os.path.join(
    ROOT, "offshore-task0-HuTingxian", "output", "task0", "turbine_coordinates.csv"
)
TURBINE_YAML = os.path.join(ROOT, "offshore-task2", "data", "iea_10MW.yaml")
TRAIN = os.path.join(OUT, "task1_training_data.csv")
HIGH_GAIN_ROSES = os.path.join(HERE, "high_gain_wind_roses.csv")

K_WAKE = 0.05
THETA = np.arange(0.0, 360.0, 5.0)
WIND_SPEEDS = np.arange(3.0, 26.0, 1.0)
YEAR = 2024
REFERENCE_SPEED = 9.0
PAIR_DEFICIT_THRESHOLD = 0.02
PAIR_DEFICIT_THRESHOLDS = (0.01, 0.02, 0.05)
NET_DEFICIT_THRESHOLD = 0.05
SEVERE_POWER_LOSS_THRESHOLD = 0.20
REPRESENTATIVE_FARMS = {2, 57, 66, 91, 126, 155, 157, 159}

with open(TURBINE_YAML, "r", encoding="utf-8") as stream:
    turbine = yaml.safe_load(stream)

ROTOR_DIAMETER = float(turbine["rotor_diameter"])
power_thrust = turbine["power_thrust_table"]
POWER_WS = np.asarray(power_thrust["wind_speed"], dtype=float)
POWER_KW = np.asarray(power_thrust["power"], dtype=float)
CT = np.asarray(power_thrust["thrust_coefficient"], dtype=float)
RATED_POWER = float(POWER_KW.max())


def power_of(speed: np.ndarray | float) -> np.ndarray:
    return np.interp(speed, POWER_WS, POWER_KW, left=0.0, right=0.0)


def ct_of(speed: np.ndarray | float) -> np.ndarray:
    return np.interp(speed, POWER_WS, CT, left=0.0, right=0.0)


def overlap_fraction(crosswind_distance: np.ndarray, wake_radius: np.ndarray) -> np.ndarray:
    """转子盘（0.5D）与尾流盘的重叠面积占转子盘面积的比例。"""
    rotor_radius = 0.5
    distance = np.abs(crosswind_distance)
    result = np.zeros_like(distance)

    full = distance <= np.abs(wake_radius - rotor_radius)
    result[full] = np.minimum(1.0, (wake_radius[full] / rotor_radius) ** 2)

    partial = (~full) & (distance < wake_radius + rotor_radius)
    if partial.any():
        d = distance[partial]
        big_r = wake_radius[partial]
        small_r = rotor_radius
        c1 = np.clip(
            (d**2 + big_r**2 - small_r**2) / (2 * d * big_r), -1.0, 1.0
        )
        c2 = np.clip(
            (d**2 + small_r**2 - big_r**2) / (2 * d * small_r), -1.0, 1.0
        )
        area = (
            big_r**2 * np.arccos(c1)
            + small_r**2 * np.arccos(c2)
            - 0.5
            * np.sqrt(
                np.maximum(
                    0.0,
                    (-d + big_r + small_r)
                    * (d + big_r - small_r)
                    * (d - big_r + small_r)
                    * (d + big_r + small_r),
                )
            )
        )
        result[partial] = area / (np.pi * small_r**2)
    return np.clip(result, 0.0, 1.0)


def pair_kernel_at_direction(
    x_d: np.ndarray, y_d: np.ndarray, theta_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回单方向的成对尾流核、顺风坐标、侧风坐标和顺风距离。"""
    angle = np.radians(theta_deg)
    along = x_d * np.cos(angle) + y_d * np.sin(angle)
    cross = -x_d * np.sin(angle) + y_d * np.cos(angle)
    downwind_distance = along[:, None] - along[None, :]
    crosswind_distance = cross[:, None] - cross[None, :]

    upstream = downwind_distance > 1e-6
    wake_radius = 0.5 + K_WAKE * np.where(upstream, downwind_distance, 0.0)
    overlap = np.where(
        upstream, overlap_fraction(crosswind_distance, wake_radius), 0.0
    )
    decay = np.where(
        upstream,
        1.0 / (1.0 + 2.0 * K_WAKE * downwind_distance) ** 2,
        0.0,
    )
    pair_kernel = decay * overlap
    return pair_kernel, along, cross, downwind_distance


def geometry_kernel(x_d: np.ndarray, y_d: np.ndarray) -> np.ndarray:
    """Q[方向, 机组]：不含 Ct 的几何尾流亏损核。"""
    result = np.zeros((len(THETA), len(x_d)))
    for index, theta in enumerate(THETA):
        pair_kernel, _, _, _ = pair_kernel_at_direction(x_d, y_d, theta)
        result[index] = np.sqrt(np.sum(pair_kernel**2, axis=1))
    return result


def weibull_weights(scale: float, shape: float) -> np.ndarray:
    weights = (
        (shape / scale)
        * (WIND_SPEEDS / scale) ** (shape - 1.0)
        * np.exp(-(WIND_SPEEDS / scale) ** shape)
    )
    return weights / weights.sum()


def power_by_direction_and_speed(q_geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回有尾流功率[方向, 风速]和无尾流功率[风速]。"""
    n_turbines = q_geometry.shape[1]
    wake_power = np.zeros((len(THETA), len(WIND_SPEEDS)))
    free_power = n_turbines * power_of(WIND_SPEEDS)
    for speed_index, speed in enumerate(WIND_SPEEDS):
        induction = 1.0 - np.sqrt(1.0 - ct_of(speed))
        total_deficit = np.clip(induction * q_geometry, 0.0, 0.95)
        effective_speed = speed * (1.0 - total_deficit)
        wake_power[:, speed_index] = power_of(effective_speed).sum(axis=1)
    return wake_power, free_power


def longest_wake_chain(edge_mask: np.ndarray, along: np.ndarray) -> int:
    """有效尾流有向图的最长路径（以机组数计）。"""
    order = np.argsort(along)
    depth = np.ones(len(along), dtype=int)
    for turbine_index in order:
        upstream_nodes = np.flatnonzero(edge_mask[turbine_index])
        if upstream_nodes.size:
            depth[turbine_index] = 1 + int(depth[upstream_nodes].max())
    return int(depth.max())


def detailed_wake_state(
    x_d: np.ndarray, y_d: np.ndarray, theta_deg: float
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """在 9 m/s 下打开尾流链、速度亏损和机组功率中间量。"""
    pair_kernel, along, cross, _ = pair_kernel_at_direction(x_d, y_d, theta_deg)
    induction = float(1.0 - np.sqrt(1.0 - ct_of(REFERENCE_SPEED)))
    pair_deficit = induction * pair_kernel
    effective_edges = pair_deficit >= PAIR_DEFICIT_THRESHOLD
    number_of_wakes = effective_edges.sum(axis=1)
    total_deficit = np.clip(
        induction * np.sqrt(np.sum(pair_kernel**2, axis=1)), 0.0, 0.95
    )
    effective_speed = REFERENCE_SPEED * (1.0 - total_deficit)
    turbine_power = power_of(effective_speed)
    free_turbine_power = float(power_of(REFERENCE_SPEED))
    relative_power_loss = 1.0 - turbine_power / free_turbine_power

    summary = {
        "theta_deg": float(theta_deg),
        "mean_deficit_pct": float(total_deficit.mean() * 100.0),
        "p90_deficit_pct": float(np.quantile(total_deficit, 0.9) * 100.0),
        "share_waked_pct": float((total_deficit >= NET_DEFICIT_THRESHOLD).mean() * 100.0),
        "mean_effective_wakes": float(number_of_wakes.mean()),
        "share_multiwake_pct": float((number_of_wakes >= 2).mean() * 100.0),
        "longest_chain_n": longest_wake_chain(effective_edges, along),
        "edge_count": int(effective_edges.sum()),
        "share_severe_power_loss_pct": float(
            (relative_power_loss >= SEVERE_POWER_LOSS_THRESHOLD).mean() * 100.0
        ),
        "farm_efficiency_9ms_pct": float(
            turbine_power.sum() / (len(x_d) * free_turbine_power) * 100.0
        ),
        "mean_power_9ms_kw": float(turbine_power.mean()),
    }
    for threshold in PAIR_DEFICIT_THRESHOLDS:
        threshold_edges = pair_deficit >= threshold
        threshold_wake_count = threshold_edges.sum(axis=1)
        suffix = f"t{int(threshold * 100):02d}"
        summary[f"mean_effective_wakes_{suffix}"] = float(
            threshold_wake_count.mean()
        )
        summary[f"share_multiwake_pct_{suffix}"] = float(
            (threshold_wake_count >= 2).mean() * 100.0
        )
        summary[f"longest_chain_n_{suffix}"] = longest_wake_chain(
            threshold_edges, along
        )
        summary[f"edge_count_{suffix}"] = int(threshold_edges.sum())
    turbines = pd.DataFrame(
        {
            "turbine_index": np.arange(len(x_d)),
            "x_D": x_d,
            "y_D": y_d,
            "alongwind_D": along,
            "crosswind_D": cross,
            "velocity_deficit_pct": total_deficit * 100.0,
            "effective_wake_count": number_of_wakes,
            "effective_speed_ms": effective_speed,
            "power_kw": turbine_power,
            "power_loss_pct": relative_power_loss * 100.0,
        }
    )
    downstream_index, upstream_index = np.where(effective_edges)
    edges = pd.DataFrame(
        {
            "upstream_turbine_index": upstream_index,
            "downstream_turbine_index": downstream_index,
            "upstream_alongwind_D": along[upstream_index],
            "upstream_crosswind_D": cross[upstream_index],
            "downstream_alongwind_D": along[downstream_index],
            "downstream_crosswind_D": cross[downstream_index],
            "pair_velocity_deficit_pct": pair_deficit[
                downstream_index, upstream_index
            ]
            * 100.0,
        }
    )
    return summary, turbines, edges


def prefix(mapping: dict, label: str) -> dict:
    return {f"{label}_{key}": value for key, value in mapping.items()}


def safe_share(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return np.nan
    return numerator / denominator * 100.0


def rose_from_direction_samples(direction_deg: np.ndarray) -> np.ndarray:
    """把逐日风向归入与 THETA 一致的 5° 扇区。"""
    nearest_index = np.floor(((direction_deg + 2.5) % 360.0) / 5.0).astype(int)
    counts = np.bincount(nearest_index, minlength=len(THETA)).astype(float)
    return counts / counts.sum()


def circular_concentration(rose: np.ndarray, order: int) -> tuple[float, float]:
    """返回第 order 阶圆统计幅度和轴角。"""
    complex_moment = np.sum(
        rose * np.exp(1j * order * np.radians(THETA))
    )
    amplitude = float(np.abs(complex_moment))
    axis_deg = float(
        (np.degrees(np.angle(complex_moment)) / order) % (360.0 / order)
    )
    return amplitude, axis_deg


def real_rose_orientation_decomposition(
    wake_power: np.ndarray,
    free_power: np.ndarray,
    speed_weights: np.ndarray,
    rose: np.ndarray,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    用真实逐日风向频率和 Weibull 风速边际分布计算旋转曲线。

    由于缺少逐日风速，风速与风向在这里按独立分布组合；该结果用于解释机制，
    不替代 FLORIS + ERA5 逐时联合分布回测。
    """
    n_direction = len(THETA)
    annual_by_rotation = np.zeros(n_direction)
    for rotation_index in range(n_direction):
        relative_index = (
            np.arange(n_direction) - rotation_index
        ) % n_direction
        climate_power_by_speed = (
            rose[:, None] * wake_power[relative_index, :]
        ).sum(axis=0)
        annual_by_rotation[rotation_index] = float(
            np.sum(speed_weights * climate_power_by_speed)
        )

    best_rotation_index = int(np.argmax(annual_by_rotation))
    mean_orientation_power = float(annual_by_rotation.mean())
    best_orientation_power = float(annual_by_rotation[best_rotation_index])
    relative_index = (
        np.arange(n_direction) - best_rotation_index
    ) % n_direction
    best_power_by_speed = (
        rose[:, None] * wake_power[relative_index, :]
    ).sum(axis=0)
    mean_orientation_power_by_speed = wake_power.mean(axis=0)
    recovered_by_speed = speed_weights * (
        best_power_by_speed - mean_orientation_power_by_speed
    )

    mean_orientation_power_by_direction = (
        wake_power.mean(axis=0)[None, :] * np.ones((n_direction, 1))
    )
    recovered_by_direction = rose * np.sum(
        speed_weights[None, :]
        * (
            wake_power[relative_index, :]
            - mean_orientation_power_by_direction
        ),
        axis=1,
    )

    total_recovery = float(recovered_by_speed.sum())
    free_annual_power = float(np.sum(speed_weights * free_power))
    summary = {
        "realrose_best_rotation_deg": float(THETA[best_rotation_index]),
        "realrose_gain_pct_of_mean": safe_share(
            best_orientation_power - mean_orientation_power,
            mean_orientation_power,
        ),
        "realrose_recovery_pp_of_free": safe_share(
            total_recovery, free_annual_power
        ),
        "realrose_top_direction_contribution_pct": safe_share(
            np.sort(recovered_by_direction)[-4:].sum(), total_recovery
        ),
    }
    for order in (1, 2, 4):
        amplitude, axis = circular_concentration(rose, order)
        summary[f"R{order}_daily"] = amplitude
        summary[f"R{order}_axis_deg"] = axis

    bins = {
        "low_3_6": WIND_SPEEDS <= 6.0,
        "partial_7_10": (WIND_SPEEDS >= 7.0) & (WIND_SPEEDS <= 10.0),
        "near_rated_11_14": (WIND_SPEEDS >= 11.0) & (WIND_SPEEDS <= 14.0),
        "saturated_15_25": WIND_SPEEDS >= 15.0,
    }
    for label, mask in bins.items():
        summary[f"realrose_recovery_share_{label}_pct"] = safe_share(
            float(recovered_by_speed[mask].sum()), total_recovery
        )

    detail = pd.DataFrame(
        {
            "theta_deg": THETA,
            "rose_probability": rose,
            "best_rotation_direction_contribution_kw": recovered_by_direction,
        }
    )
    speed_detail = pd.DataFrame(
        {
            "ws_bin_ms": WIND_SPEEDS,
            "weibull_weight": speed_weights,
            "best_orientation_power_kw": best_power_by_speed,
            "mean_orientation_power_kw": mean_orientation_power_by_speed,
            "weighted_recovered_power_kw": recovered_by_speed,
            "recovery_contribution_pct": np.divide(
                recovered_by_speed,
                total_recovery,
                out=np.zeros_like(recovered_by_speed),
                where=abs(total_recovery) > 1e-12,
            )
            * 100.0,
        }
    )
    return summary, detail, speed_detail


print("加载数据 ...")
coordinates = pd.read_csv(COORD)
coordinates = coordinates[coordinates["year"] == YEAR]
training = pd.read_csv(TRAIN)
training_index = training.set_index("farm_id")
if os.path.exists(HIGH_GAIN_ROSES):
    high_gain_roses = pd.read_csv(HIGH_GAIN_ROSES)
else:
    high_gain_roses = pd.DataFrame(columns=["farm_id", "wd_deg"])

weibull_scale_median = float(training["weibull_A"].median())
weibull_shape_median = float(training["weibull_k"].median())

farm_rows: list[dict] = []
speed_rows: list[pd.DataFrame] = []
turbine_rows: list[pd.DataFrame] = []
edge_rows: list[pd.DataFrame] = []
rose_rows: list[pd.DataFrame] = []
rose_speed_rows: list[pd.DataFrame] = []

start_time = time.time()
for farm_counter, (farm_id, farm_coordinates) in enumerate(
    coordinates.groupby("farm_id"), start=1
):
    if len(farm_coordinates) < 2:
        continue

    x_d = (
        farm_coordinates["x_m"].to_numpy() - farm_coordinates["x_m"].mean()
    ) / ROTOR_DIAMETER
    y_d = (
        farm_coordinates["y_m"].to_numpy() - farm_coordinates["y_m"].mean()
    ) / ROTOR_DIAMETER

    if farm_id in training_index.index:
        training_row = training_index.loc[farm_id]
        scale = float(training_row.get("weibull_A", weibull_scale_median))
        shape = float(training_row.get("weibull_k", weibull_shape_median))
    else:
        training_row = None
        scale = weibull_scale_median
        shape = weibull_shape_median

    q_geometry = geometry_kernel(x_d, y_d)
    wake_power, free_power = power_by_direction_and_speed(q_geometry)
    weights = weibull_weights(scale, shape)
    free_annual_power = float(np.sum(weights * free_power))
    wake_annual_power = np.sum(wake_power * weights[None, :], axis=1)
    energy_loss = 1.0 - wake_annual_power / free_annual_power

    bad_index = int(np.argmax(energy_loss))
    good_index = int(np.argmin(energy_loss))
    bad_theta = float(THETA[bad_index])
    good_theta = float(THETA[good_index])

    bad_state, bad_turbines, bad_edges = detailed_wake_state(
        x_d, y_d, bad_theta
    )
    good_state, good_turbines, good_edges = detailed_wake_state(
        x_d, y_d, good_theta
    )

    bad_power = wake_power[bad_index]
    good_power = wake_power[good_index]
    recovered_power = good_power - bad_power
    recovered_energy = weights * recovered_power
    total_recovered_energy = float(recovered_energy.sum())
    bad_annual_power = float(np.sum(weights * bad_power))

    speed_groups = {
        "low_3_6": WIND_SPEEDS <= 6.0,
        "partial_7_10": (WIND_SPEEDS >= 7.0) & (WIND_SPEEDS <= 10.0),
        "near_rated_11_14": (WIND_SPEEDS >= 11.0) & (WIND_SPEEDS <= 14.0),
        "saturated_15_25": WIND_SPEEDS >= 15.0,
    }

    row = {
        "farm_id": int(farm_id),
        "n_turb": int(len(farm_coordinates)),
        "weibull_A": scale,
        "weibull_k": shape,
        "theta_bad_deg": bad_theta,
        "theta_good_deg": good_theta,
        "direction_separation_deg": float(
            min((good_theta - bad_theta) % 180.0, (bad_theta - good_theta) % 180.0)
        ),
        "Lenergy_bad_pct": float(energy_loss[bad_index] * 100.0),
        "Lenergy_good_pct": float(energy_loss[good_index] * 100.0),
        "delta_Lenergy_pp": float(
            (energy_loss[bad_index] - energy_loss[good_index]) * 100.0
        ),
        "annual_recovery_pct_of_bad": safe_share(
            total_recovered_energy, bad_annual_power
        ),
        "annual_recovery_pp_of_free": safe_share(
            total_recovered_energy, free_annual_power
        ),
        "peak_recovery_speed_ms": float(
            WIND_SPEEDS[int(np.argmax(recovered_energy))]
        ),
    }
    row.update(prefix(bad_state, "bad"))
    row.update(prefix(good_state, "good"))

    for metric in [
        "mean_deficit_pct",
        "p90_deficit_pct",
        "share_waked_pct",
        "mean_effective_wakes",
        "share_multiwake_pct",
        "longest_chain_n",
        "edge_count",
        "share_severe_power_loss_pct",
        "farm_efficiency_9ms_pct",
        "mean_power_9ms_kw",
        "mean_effective_wakes_t01",
        "share_multiwake_pct_t01",
        "longest_chain_n_t01",
        "edge_count_t01",
        "mean_effective_wakes_t02",
        "share_multiwake_pct_t02",
        "longest_chain_n_t02",
        "edge_count_t02",
        "mean_effective_wakes_t05",
        "share_multiwake_pct_t05",
        "longest_chain_n_t05",
        "edge_count_t05",
    ]:
        row[f"delta_{metric}"] = bad_state[metric] - good_state[metric]

    for group_name, mask in speed_groups.items():
        group_recovery = float(recovered_energy[mask].sum())
        group_free_energy = float((weights * free_power)[mask].sum())
        row[f"recovery_share_{group_name}_pct"] = safe_share(
            group_recovery, total_recovered_energy
        )
        row[f"recovery_pp_free_{group_name}"] = safe_share(
            group_recovery, free_annual_power
        )
        row[f"free_energy_share_{group_name}_pct"] = safe_share(
            group_free_energy, free_annual_power
        )

    if training_row is not None:
        for column in [
            "A",
            "wake_pool",
            "WCI",
            "spacing_D",
            "aspect_ratio",
            "pc1_share",
            "ws_mean",
            "ws_std",
            "frac_below_rated",
            "wd_entropy_norm",
            "lat",
            "lon",
            "country",
        ]:
            row[column] = training_row.get(column, np.nan)

    rose_samples = high_gain_roses[
        high_gain_roses["farm_id"] == int(farm_id)
    ]
    if len(rose_samples):
        actual_rose = rose_from_direction_samples(
            rose_samples["wd_deg"].to_numpy(dtype=float)
        )
        (
            rose_summary,
            rose_detail,
            rose_speed_detail,
        ) = real_rose_orientation_decomposition(
            wake_power, free_power, weights, actual_rose
        )
        row.update(rose_summary)
        rose_detail.insert(0, "farm_id", int(farm_id))
        rose_rows.append(rose_detail)
        rose_speed_detail.insert(0, "farm_id", int(farm_id))
        rose_speed_rows.append(rose_speed_detail)

    farm_rows.append(row)

    speed_frame = pd.DataFrame(
        {
            "farm_id": int(farm_id),
            "ws_bin_ms": WIND_SPEEDS,
            "weibull_weight": weights,
            "free_power_kw": free_power,
            "bad_power_kw": bad_power,
            "good_power_kw": good_power,
            "recovered_power_kw": recovered_power,
            "weighted_recovered_power_kw": recovered_energy,
            "recovery_pct_of_free_at_ws": np.divide(
                recovered_power,
                free_power,
                out=np.zeros_like(recovered_power),
                where=free_power > 0,
            )
            * 100.0,
        }
    )
    speed_rows.append(speed_frame)

    if int(farm_id) in REPRESENTATIVE_FARMS:
        for state_name, state_theta, turbine_frame, edge_frame in [
            ("bad", bad_theta, bad_turbines, bad_edges),
            ("good", good_theta, good_turbines, good_edges),
        ]:
            turbine_frame.insert(0, "state", state_name)
            turbine_frame.insert(0, "theta_deg", state_theta)
            turbine_frame.insert(0, "farm_id", int(farm_id))
            turbine_rows.append(turbine_frame)
            edge_frame.insert(0, "state", state_name)
            edge_frame.insert(0, "theta_deg", state_theta)
            edge_frame.insert(0, "farm_id", int(farm_id))
            edge_rows.append(edge_frame)

    if farm_counter % 20 == 0:
        elapsed = time.time() - start_time
        print(
            f"  [{farm_counter}/{coordinates.farm_id.nunique()}] "
            f"{elapsed:.1f}s"
        )

farm_metrics = pd.DataFrame(farm_rows)
speed_detail = pd.concat(speed_rows, ignore_index=True)
turbine_detail = pd.concat(turbine_rows, ignore_index=True)
edge_detail = pd.concat(edge_rows, ignore_index=True)
rose_detail = (
    pd.concat(rose_rows, ignore_index=True)
    if rose_rows
    else pd.DataFrame()
)
rose_speed_detail = (
    pd.concat(rose_speed_rows, ignore_index=True)
    if rose_speed_rows
    else pd.DataFrame()
)

farm_output = os.path.join(OUT, "mechanism_dual_layer_farms.csv")
speed_output = os.path.join(OUT, "mechanism_dual_layer_speed.csv")
turbine_output = os.path.join(OUT, "mechanism_dual_layer_turbines.csv")
edge_output = os.path.join(OUT, "mechanism_dual_layer_edges.csv")
summary_output = os.path.join(OUT, "mechanism_dual_layer_summary.md")
rose_output = os.path.join(OUT, "mechanism_dual_layer_realrose.csv")
rose_speed_output = os.path.join(
    OUT, "mechanism_dual_layer_realrose_speed.csv"
)

farm_metrics.to_csv(farm_output, index=False, encoding="utf-8-sig")
speed_detail.to_csv(speed_output, index=False, encoding="utf-8-sig")
turbine_detail.to_csv(turbine_output, index=False, encoding="utf-8-sig")
edge_detail.to_csv(edge_output, index=False, encoding="utf-8-sig")
rose_detail.to_csv(rose_output, index=False, encoding="utf-8-sig")
rose_speed_detail.to_csv(
    rose_speed_output, index=False, encoding="utf-8-sig"
)


def rho_with_a(column: str) -> tuple[float, float, int]:
    subset = farm_metrics[["A", column]].dropna()
    result = spearmanr(subset["A"], subset[column])
    return float(result.statistic), float(result.pvalue), len(subset)


correlation_columns = [
    "delta_mean_deficit_pct",
    "delta_p90_deficit_pct",
    "delta_share_multiwake_pct",
    "delta_longest_chain_n",
    "delta_share_severe_power_loss_pct",
    "annual_recovery_pp_of_free",
]

high_group = farm_metrics[farm_metrics["A"] > 5.2]
low_group = farm_metrics[farm_metrics["A"] <= 2.0]

summary_lines = [
    "# 双层机制重算结果",
    "",
    f"- 样本：{len(farm_metrics)} 个终期真实排布。",
    f"- 参考尾流状态：{REFERENCE_SPEED:.0f} m/s；有效单尾流阈值："
    f"{PAIR_DEFICIT_THRESHOLD*100:.0f}% 速度亏损。",
    "",
    "## 与 FLORIS + ERA5 朝向幅度 A 的秩相关",
    "",
    "| 机制量 | Spearman ρ | p | n |",
    "|---|---:|---:|---:|",
]
for column in correlation_columns:
    rho, p_value, n_sample = rho_with_a(column)
    summary_lines.append(
        f"| {column} | {rho:.3f} | {p_value:.2e} | {n_sample} |"
    )

summary_lines.extend(
    [
        "",
        "## 高值场与低值场的中位对照",
        "",
        "| 指标 | A>5.2% | A≤2% |",
        "|---|---:|---:|",
    ]
)
comparison_columns = [
    "delta_mean_deficit_pct",
    "delta_share_multiwake_pct",
    "delta_longest_chain_n",
    "delta_share_severe_power_loss_pct",
    "annual_recovery_pp_of_free",
    "recovery_share_partial_7_10_pct",
    "recovery_share_near_rated_11_14_pct",
]
for column in comparison_columns:
    summary_lines.append(
        f"| {column} | {high_group[column].median():.2f} | "
        f"{low_group[column].median():.2f} |"
    )

summary_lines.extend(
    [
        "",
        "## 尾流阈值敏感性",
        "",
        "| 单尾流速度亏损阈值 | 多重尾流比例下降 vs A：ρ | 最长链缩短 vs A：ρ |",
        "|---:|---:|---:|",
    ]
)
for threshold in PAIR_DEFICIT_THRESHOLDS:
    suffix = f"t{int(threshold * 100):02d}"
    rho_multi, _, _ = rho_with_a(f"delta_share_multiwake_pct_{suffix}")
    rho_chain, _, _ = rho_with_a(f"delta_longest_chain_n_{suffix}")
    summary_lines.append(
        f"| {threshold*100:.0f}% | {rho_multi:.3f} | {rho_chain:.3f} |"
    )

if farm_metrics["realrose_gain_pct_of_mean"].notna().any():
    rose_subset = farm_metrics[
        ["A", "realrose_gain_pct_of_mean"]
    ].dropna()
    rose_rho = spearmanr(
        rose_subset["A"], rose_subset["realrose_gain_pct_of_mean"]
    )
    rose_mae = float(
        np.mean(
            np.abs(
                rose_subset["A"]
                - rose_subset["realrose_gain_pct_of_mean"]
            )
        )
    )
    summary_lines.extend(
        [
            "",
            "## 7 场真实 30 年逐日风向频率 × Weibull 风速分解",
            "",
            f"- 机制前向预测与 FLORIS + ERA5 幅度的 Spearman ρ = "
            f"{rose_rho.statistic:.3f}（p={rose_rho.pvalue:.3g}），"
            f"平均绝对误差为 {rose_mae:.2f} 个百分点。",
            "",
            "| farm | A_FLORIS (%) | R1 | R2 | 机制预测增益 (%) | "
            "3--10 m/s贡献 (%) | 11--14 m/s贡献 (%) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, item in farm_metrics[
        farm_metrics["realrose_gain_pct_of_mean"].notna()
    ].sort_values("farm_id").iterrows():
        below_rated_share = (
            item["realrose_recovery_share_low_3_6_pct"]
            + item["realrose_recovery_share_partial_7_10_pct"]
        )
        summary_lines.append(
            f"| {int(item.farm_id)} | {item.A:.2f} | "
            f"{item.R1_daily:.3f} | {item.R2_daily:.3f} | "
            f"{item.realrose_gain_pct_of_mean:.2f} | "
            f"{below_rated_share:.1f} | "
            f"{item.realrose_recovery_share_near_rated_11_14_pct:.1f} |"
        )

summary_lines.extend(
    [
        "",
        "## 代表场",
        "",
        "| farm | country | A (%) | 尾流链缩短（台） | 多重尾流比例下降（pp） | "
        "平均速度亏损下降（pp） | 年能量恢复/无尾流AEP（pp） | 峰值贡献风速（m/s） |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
)
for _, representative in farm_metrics[
    farm_metrics["farm_id"].isin(REPRESENTATIVE_FARMS)
].sort_values("farm_id").iterrows():
    summary_lines.append(
        f"| {int(representative.farm_id)} | {representative.get('country','')} | "
        f"{representative.get('A',np.nan):.2f} | "
        f"{representative.delta_longest_chain_n:.0f} | "
        f"{representative.delta_share_multiwake_pct:.1f} | "
        f"{representative.delta_mean_deficit_pct:.1f} | "
        f"{representative.annual_recovery_pp_of_free:.2f} | "
        f"{representative.peak_recovery_speed_ms:.0f} |"
    )

with open(summary_output, "w", encoding="utf-8") as stream:
    stream.write("\n".join(summary_lines) + "\n")

print(f"\n完成 {len(farm_metrics)} 场，用时 {time.time()-start_time:.1f}s")
print(f"  -> {os.path.relpath(farm_output, ROOT)}")
print(f"  -> {os.path.relpath(speed_output, ROOT)}")
print(f"  -> {os.path.relpath(turbine_output, ROOT)}")
print(f"  -> {os.path.relpath(edge_output, ROOT)}")
print(f"  -> {os.path.relpath(summary_output, ROOT)}")
print(f"  -> {os.path.relpath(rose_output, ROOT)}")
print(f"  -> {os.path.relpath(rose_speed_output, ROOT)}")
print("\n".join(summary_lines[:25]))
