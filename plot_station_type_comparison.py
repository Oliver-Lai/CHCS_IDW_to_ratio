from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "1_filtered_output"
META_PATH = ROOT / "空氣品質監測站基本資料.csv"
OUT_DIR = ROOT / "station_type_comparison"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 中文字型設定
for font_path in [
    ROOT / "NotoSansCJKtc-Regular.otf",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
]:
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Noto Sans CJK TC"
        break
else:
    plt.rcParams["font.family"] = "sans-serif"

FACTORS = ["AMB_TEMP", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5", "RH", "SO2"]


def normalize_name(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().replace("台", "臺")
    return x


def load_station_metadata():
    meta = pd.read_csv(META_PATH, encoding="utf-8-sig")
    meta["sitename_norm"] = meta["sitename"].apply(normalize_name)
    meta["sitetype"] = meta["sitetype"].fillna("")
    return meta


def collect_station_data(meta):
    records = []

    for path in sorted(DATA_ROOT.rglob("*.csv")):
        if "2_周統計_平均值_" not in path.name:
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")
        if "日期" not in df.columns or "地區" not in df.columns:
            continue

        # 先用檔名中的站名嘗試解析
        station_name = ""
        m = re.search(r"2_周統計_平均值_(.+)_(\d{4})\.csv$", path.name)
        if m:
            station_name = normalize_name(m.group(1))
        else:
            station_name = normalize_name(df["地區"].iloc[0]) if not df.empty else ""

        if not station_name:
            continue

        meta_row = meta[meta["sitename_norm"] == station_name]
        if meta_row.empty:
            continue

        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"]).copy()
        if df.empty:
            continue

        df["站名"] = station_name
        df["站點類型"] = meta_row.iloc[0]["sitetype"]
        records.append(df)

    if not records:
        raise FileNotFoundError("沒有找到可處理的站點資料檔")

    all_df = pd.concat(records, ignore_index=True)
    return all_df


def compute_type_daily_mean(all_df):
    # 依照站點類型與日期計算各因子的平均值
    daily_type_mean = []
    for sitetype in sorted(all_df["站點類型"].dropna().unique()):
        sub = all_df[all_df["站點類型"] == sitetype].copy()
        if sub.empty:
            continue
        for factor in FACTORS:
            if factor not in sub.columns:
                continue
            temp = sub[["日期", factor]].copy()
            temp = temp.dropna()
            if temp.empty:
                continue
            temp = temp.groupby("日期")[factor].mean().reset_index()
            temp.columns = ["日期", "值"]
            temp["因子"] = factor
            temp["站點類型"] = sitetype
            daily_type_mean.append(temp)

    return pd.concat(daily_type_mean, ignore_index=True)


def save_summary_csv(all_df):
    summary = all_df.groupby("站點類型")[FACTORS].mean().reset_index()
    summary.to_csv(OUT_DIR / "site_type_factor_means.csv", index=False, encoding="utf-8-sig")
    print("已輸出摘要 CSV:", OUT_DIR / "site_type_factor_means.csv")


def plot_type_comparison(daily_type_mean):
    special_types = sorted(set(daily_type_mean["站點類型"]) - {"一般站"})
    general = daily_type_mean[daily_type_mean["站點類型"] == "一般站"].copy()

    for sitetype in special_types:
        special = daily_type_mean[daily_type_mean["站點類型"] == sitetype].copy()
        if special.empty:
            continue

        fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
        axes = axes.ravel()

        for ax, factor in zip(axes, FACTORS):
            g = general[general["因子"] == factor][["日期", "值"]].rename(columns={"值": "一般站平均"})
            s = special[special["因子"] == factor][["日期", "值"]].rename(columns={"值": f"{sitetype}平均"})

            if g.empty or s.empty:
                ax.set_title(factor)
                ax.text(0.5, 0.5, "無資料", ha="center", va="center")
                ax.set_axis_off()
                continue

            merged = g.merge(s, on="日期", how="outer")
            merged = merged.sort_values("日期")

            ax.plot(merged["日期"], merged["一般站平均"], color="black", linestyle="--", linewidth=2, label="一般站平均")
            ax.plot(merged["日期"], merged[f"{sitetype}平均"], color="tab:red", linewidth=1.8, label=f"{sitetype}平均")
            ax.set_title(factor)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"{sitetype} 與一般站平均比較", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = PLOT_DIR / f"{sitetype}_vs_general.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("已輸出圖:", out_path)


def main():
    meta = load_station_metadata()
    all_df = collect_station_data(meta)
    save_summary_csv(all_df)

    daily_type_mean = compute_type_daily_mean(all_df)
    plot_type_comparison(daily_type_mean)

    print("分析完成，結果輸出至:", OUT_DIR)


if __name__ == "__main__":
    main()