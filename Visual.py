import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def STATS(Data, Agents):
    data = Data.copy()
    agent_results = [agent.get_result() for agent in Agents]

    rows = []
    for res in agent_results:
        r = pd.Series(res["return"], dtype=float)
        if len(data) > 0:
            r = r.reindex(range(len(data))).fillna(0.0)
        r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        n = len(r)
        mean_r = r.mean() if n > 0 else 0.0
        std_r = r.std(ddof=1) if n > 1 else 0.0
        downside = r[r < 0]
        downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0

        # Calculate Transaction-based Win Rate
        pos_state = pd.Series(res.get("position_state_for_pnl", []), dtype=float)
        
        if len(pos_state) != len(r) or len(pos_state) == 0:
            # Fallback to daily win rate if pos_state is missing/malformed
            trading_days = r[r != 0]
            win_rate = (trading_days > 0).mean() if len(trading_days) > 0 else 0.0
        else:
            state_changes = pos_state.diff().ne(0).cumsum()
            trade_returns = []
            for _, group in r.groupby(state_changes):
                state = pos_state.loc[group.index[0]]
                if state != 0:
                    # r is log return. sum of log returns = log of product of (1+R)
                    # A transaction is profitable if the product of (1+R) > 1, which means sum(r) > 0
                    trade_returns.append(group.sum())
            
            wins = sum(1 for ret in trade_returns if ret > 0)
            win_rate = wins / len(trade_returns) if len(trade_returns) > 0 else 0.0

        # r is log return; aggregate in log space then convert back.
        cum_log = r.cumsum()
        equity = np.exp(cum_log)
        total_return = np.exp(cum_log.iloc[-1]) - 1.0 if n > 0 else 0.0
        annual_return = np.exp(mean_r * 252.0) - 1.0 if n > 0 else 0.0
        annual_vol = std_r * np.sqrt(252.0) if std_r > 0 else 0.0

        sharpe = (mean_r / std_r) * np.sqrt(252.0) if std_r > 0 else np.nan
        sortino = (mean_r / downside_std) * np.sqrt(252.0) if downside_std > 0 else np.nan

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = drawdown.min() if n > 0 else 0.0

        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

        var_r = r.var(ddof=1) if n > 1 else 0.0
        kelly = mean_r / var_r if var_r > 0 else np.nan

        rows.append(
            {
                "Agent": res["display_name"],
                "Win Rate": win_rate,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Annual Return": annual_return,
                "Annual Volatility": annual_vol,
                "Max Drawdown": max_drawdown,
                "Calmar Ratio": calmar,
                "Kelly's Criteria": kelly,
            }
        )

    stats_df = pd.DataFrame(rows).set_index("Agent")

    # Use reset_index for printing so "Agent" is a column and aligns with headers on one row
    display_df = stats_df.reset_index()
    pct_cols = ["Win Rate", "Annual Return", "Annual Volatility", "Max Drawdown"]
    ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Kelly's Criteria"]

    for col in pct_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NaN")
    for col in ratio_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")

    print("\nStrategy Statistics")
    print("-" * 120)
    print(
        display_df.to_string(
            index=False,
            col_space=14,
            justify="center",
        )
    )
    print("-" * 120)
    return stats_df

def Visuallize_Result(Data, Agents):
    data = Data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    # Underlying value ratio: current value / initial value
    init_spot = float(data["Stock_Close"].iloc[0]) if len(data) else np.nan
    if np.isfinite(init_spot) and init_spot != 0:
        underlying_value_ratio = data["Stock_Close"] / init_spot
    else:
        underlying_value_ratio = pd.Series(np.nan, index=data.index)

    # Collect all agent results once
    agent_results = []
    for agent in Agents:
        r = agent.get_result()
        agent_results.append(r)

    fig = plt.figure(figsize=(18, 14))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.8, 1.35, 1.8], hspace=0.28)

    # -------------------------------------------------
    # 1) Cumulated return: all agents + underlying
    # -------------------------------------------------
    ax1 = fig.add_subplot(outer[0, 0])
    ax1.plot(data["Date"], underlying_value_ratio, label="Underlying", linewidth=2.0, color="black")

    for res in agent_results:
        daily_ret = pd.Series(res["return"], index=data.index, dtype=float).fillna(0.0)
        # Cumulated value ratio from log returns: exp(sum(log_return))
        value_ratio = np.exp(daily_ret.cumsum())
        ax1.plot(data["Date"], value_ratio, label=res["display_name"], linewidth=1.8)

    ax1.set_title("Cumulated Return", fontsize=15)
    ax1.set_ylabel("Value Ratio", fontsize=15)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.grid(alpha=0.30)
    ncol_top = min(4, max(2, 2 + len(agent_results)))
    ax1.legend(
        loc="upper left",
        fontsize=11,
        ncol=ncol_top,
        framealpha=0.92,
    )

    # -------------------------------------------------
    # 2) Daily return (line only)
    # -------------------------------------------------
    ax2 = fig.add_subplot(outer[1, 0], sharex=ax1)

    all_ret = []
    for res in agent_results:
        daily_ret = pd.Series(res["return"], index=data.index, dtype=float).fillna(0.0)
        all_ret.append(daily_ret.values)
        ax2.plot(data["Date"], daily_ret, label=res["display_name"], linewidth=1.2)

    # Symmetric limits around 0 for return axis
    ret_flat = np.concatenate(all_ret) if all_ret else np.array([0.0])
    ret_max = float(np.nanmax(np.abs(ret_flat))) if np.any(np.isfinite(ret_flat)) else 0.0
    ret_lim = max(ret_max * 1.05, 1e-9)
    ax2.set_ylim(-ret_lim, ret_lim)

    ax2.axhline(0.0, color="gray", linewidth=1.0, alpha=0.8)
    ax2.set_title("Daily Return (line)", fontsize=15)
    ax2.set_ylabel("Return Rate", fontsize=15)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.grid(alpha=0.30)
    ax2.legend(loc="upper left", fontsize=11, framealpha=0.92, ncol=min(3, max(1, len(agent_results))))
    ax2.tick_params(axis="x", labelbottom=False)

    # -------------------------------------------------
    # 3) Realized vs implied vol (levels)
    # -------------------------------------------------
    ax_rv_iv = fig.add_subplot(outer[2, 0], sharex=ax1)
    rv_series = pd.to_numeric(data.get("RV", pd.Series(index=data.index, dtype=float)), errors="coerce")
    iv_series = pd.to_numeric(
        data.get("Straddle_imp_vol", pd.Series(index=data.index, dtype=float)),
        errors="coerce",
    )
    ax_rv_iv.plot(data["Date"], rv_series, label="RV", linewidth=1.4, color="#2ca02c")
    ax_rv_iv.plot(data["Date"], iv_series, label="IV (straddle)", linewidth=1.4, color="#d62728")
    ax_rv_iv.set_title("Realized vs implied volatility", fontsize=15)
    ax_rv_iv.set_ylabel("Volatility", fontsize=15)
    ax_rv_iv.tick_params(axis="both", labelsize=13)
    ax_rv_iv.grid(alpha=0.30)
    ax_rv_iv.legend(loc="upper left", fontsize=11, framealpha=0.92, ncol=2)
    ax_rv_iv.tick_params(axis="x", labelbottom=False)

    # -------------------------------------------------
    # 4) VRP signal panel: VRP, mean, mean +/- std
    # -------------------------------------------------
    ax3 = fig.add_subplot(outer[3, 0], sharex=ax1)
    vrp = pd.to_numeric(data.get("VRP", pd.Series(index=data.index, dtype=float)), errors="coerce")
    if "VRP_20d_mean" in data.columns:
        vrp_mean = pd.to_numeric(data["VRP_20d_mean"], errors="coerce")
    else:
        vrp_mean = vrp.rolling(window=20, min_periods=20).mean()
    if "VRP_20d_std" in data.columns:
        vrp_std = pd.to_numeric(data["VRP_20d_std"], errors="coerce")
    else:
        vrp_std = vrp.rolling(window=20, min_periods=20).std()

    upper_1s = vrp_mean + vrp_std
    lower_1s = vrp_mean - vrp_std

    mean_band_color = "#ff7f0e"
    ax3.plot(data["Date"], vrp, label="VRP", linewidth=1.3, color="#1f77b4")
    ax3.plot(data["Date"], vrp_mean, label="VRP Mean", linewidth=1.3, color=mean_band_color)
    ax3.plot(
        data["Date"],
        upper_1s,
        label="VRP Mean + 1*Std",
        linewidth=1.1,
        linestyle="--",
        color=mean_band_color,
    )
    ax3.plot(
        data["Date"],
        lower_1s,
        label="VRP Mean - 1*Std",
        linewidth=1.1,
        linestyle=":",
        color=mean_band_color,
    )
    ax3.axhline(0.0, color="gray", linewidth=0.9, alpha=0.7)
    ax3.set_title("VRP Signal (Level, Mean, and +/- 1 Std)", fontsize=15)
    ax3.set_ylabel("VRP", fontsize=15)
    ax3.tick_params(axis="both", labelsize=13)
    ax3.grid(alpha=0.30)
    ax3.legend(loc="upper left", fontsize=11, ncol=2, framealpha=0.92)

    ax1.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="x", rotation=20)
    fig.subplots_adjust(hspace=0.28, bottom=0.10)
    plt.show()

def Visuallize_Greeks_exposure(Data, Agents):
    data = Data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    agent_results = []
    for agent in Agents:
        agent_results.append(agent.get_result())

    greek_names = ["delta", "gamma", "vega", "theta", "vanna", "volga", "rho"]
    n_g = len(greek_names)
    # Hard-coded vertical spacing (GridSpec hspace is a fraction of average subplot height).
    # Nested header: small gap title↔legend; outer: modest gap header block↔first panel;
    # inner plots: larger gap so panel titles / x-labels do not overlap.
    HSPACE_TITLE_LEGEND = 0.0
    HSPACE_HEADER_PLOTS = 0.06
    HSPACE_GREEK_PANELS = 0.32
    RATIO_TITLE_ROW = 0.24
    RATIO_LEGEND_ROW = 1.0
    RATIO_HEADER_VS_PLOTS = 0.40

    fig = plt.figure(figsize=(18, 33))
    gs_outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[RATIO_HEADER_VS_PLOTS, 10.0],
        hspace=HSPACE_HEADER_PLOTS,
        left=0.06,
        right=0.98,
        top=0.99,
        bottom=0.035,
    )
    gs_head = gs_outer[0].subgridspec(
        2,
        1,
        height_ratios=[RATIO_TITLE_ROW, RATIO_LEGEND_ROW],
        hspace=HSPACE_TITLE_LEGEND,
    )
    ax_title = fig.add_subplot(gs_head[0, 0])
    ax_title.set_axis_off()
    ax_title.text(
        0.5,
        0.75,
        "Greeks Exposure",
        ha="center",
        va="center",
        fontsize=19,
        transform=ax_title.transAxes,
    )
    ax_legend = fig.add_subplot(gs_head[1, 0])
    ax_legend.set_axis_off()

    gs_plots = gs_outer[1].subgridspec(
        n_g,
        1,
        height_ratios=[3.2] * n_g,
        hspace=HSPACE_GREEK_PANELS,
    )
    axes = []
    for i in range(n_g):
        ax = fig.add_subplot(gs_plots[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)

    for idx, greek in enumerate(greek_names):
        ax = axes[idx]
        for res in agent_results:
            g_series = pd.Series(
                res["greeks_attribute"][greek],
                index=data.index,
                dtype=float,
            ).fillna(0.0)
            ax.plot(
                data["Date"],
                g_series,
                label=res["display_name"] if idx == 0 else "_nolegend_",
                linewidth=1.1
            )

            if greek == "delta":
                delta_exp = pd.Series(res["actual_delta"], index=data.index, dtype=float).fillna(0.0)
                ax.plot(
                    data["Date"],
                    delta_exp,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"{res['display_name']} actual_delta",
                )

        ax.axhline(0.0, color="gray", linewidth=0.9, alpha=0.8)
        # pad is in points; values <<1 (e.g. 0.1) collapse title into the plot / next panel
        ax.set_title(f"{greek.capitalize()} Exposure", pad=12, fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.30)

    # Share x: only bottom axis shows date ticks (avoids overlap + duplicate labels)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    # No horizontal padding: tight x-limits in numeric date space + lock autoscale
    dates = pd.to_datetime(data["Date"], errors="coerce")
    x0, x1 = dates.min(), dates.max()
    for ax in axes:
        ax.margins(x=0)
    if pd.notna(x0) and pd.notna(x1):
        x_left = mdates.date2num(x0)
        x_right = mdates.date2num(x1)
        axes[-1].set_xlim(x_left, x_right)
        axes[-1].set_autoscalex_on(False)
        bottom_ax = axes[-1]
        bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        bottom_ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(bottom_ax.xaxis.get_major_locator())
        )

    axes[-1].tick_params(axis="x", rotation=20, labelsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    n_lbl = len(labels)
    legend_ncol = max(1, int(np.ceil(n_lbl / 2))) if n_lbl else 1

    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=legend_ncol,
        fontsize=14,
        framealpha=0.95,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.9,
        borderpad=0.35,
    )
    plt.show()

def Visuallize_Greeks_Attribute(Data, Agents):
    data = Data.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    agent_results = []
    for agent in Agents:
        agent_results.append(agent.get_result())

    greek_names = ["delta", "gamma", "vega", "theta", "vanna", "volga", "rho", "residual"]
    fig, axes = plt.subplots(
        len(greek_names),
        1,
        figsize=(18, 22),
        sharex=True,
    )
    fig.suptitle("Greeks Attribution (in Dollar)", fontsize=19, y=0.997)

    for idx, greek in enumerate(greek_names):
        ax = axes[idx]
        for res in agent_results:
            attr_series = pd.Series(
                res["greeks_attribute"][greek],
                index=data.index,
                dtype=float,
            ).fillna(0.0)
            ax.plot(
                data["Date"],
                attr_series,
                label=res["display_name"] if idx == 0 else "_nolegend_",
                linewidth=1.1,
            )

        ax.axhline(0.0, color="gray", linewidth=0.9, alpha=0.8)
        ax.set_title(f"{greek.capitalize()} Attribution", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.30)

    axes[-1].tick_params(axis="x", rotation=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=min(6, max(1, len(labels))),
        fontsize=15,
        framealpha=0.95,
        borderaxespad=0.05,
    )
    fig.subplots_adjust(hspace=0.35, top=0.932, bottom=0.04)
    plt.show()

def Greeks_Attribution_Pie(Data, Agents):
    agent_results = [agent.get_result() for agent in Agents]
    if len(agent_results) == 0:
        print("No agents provided.")
        return

    greek_names = ["delta", "gamma", "vega", "theta", "vanna", "volga", "rho", "residual"]
    colors = ["#1f77b4", "#9bb2d1", "#ff7f0e", "#f1b97a", "#2ca02c", "#98df8a", "#9467bd", "#d62728"]

    n_agents = len(agent_results)
    # Keep a consistent 2-column layout (two subplots per row).
    ncols = 1 if n_agents == 1 else 2
    nrows = int(np.ceil(n_agents / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5.6 * nrows),
    )

    axes = np.array(axes).reshape(-1)
    for i, res in enumerate(agent_results):
        ax = axes[i]
        vals = np.array(
            [
                np.nansum(pd.Series(res["greeks_attribute"][g], dtype=float))
                for g in greek_names
            ],
            dtype=float,
        )

        total_abs = np.sum(np.abs(vals))
        if total_abs <= 1e-12:
            ax.text(0.5, 0.5, "No attribution data", ha="center", va="center", fontsize=15)
            ax.set_title(res["display_name"], fontsize=16)
            ax.axis("off")
            continue

        sizes = np.abs(vals)
        pct_signed = vals / total_abs * 100.0
        pct_labels = [f"{x:+.1f}%" for x in pct_signed]

        wedges, _texts = ax.pie(
            sizes,
            labels=None,
            colors=colors,
            startangle=90,
            textprops={"fontsize": 13},
        )

        # Place component labels + signed percentages near slices with collision avoidance.
        labels = [f"{g.capitalize()} {p}" for g, p in zip(greek_names, pct_labels)]
        label_points = []
        for wedge, lbl in zip(wedges, labels):
            ang = 0.5 * (wedge.theta1 + wedge.theta2)
            rad = np.deg2rad(ang)
            x, y = np.cos(rad), np.sin(rad)
            side = "right" if x >= 0 else "left"
            label_points.append({"x": x, "y": y, "side": side, "label": lbl})

        def _spread(points, min_gap=0.14):
            points_sorted = sorted(points, key=lambda p: p["y"])
            last_y = -10.0
            for p in points_sorted:
                p["y_adj"] = max(p["y"], last_y + min_gap)
                last_y = p["y_adj"]
            # keep inside visible area
            ymax = max((p["y_adj"] for p in points_sorted), default=1.0)
            ymin = min((p["y_adj"] for p in points_sorted), default=-1.0)
            if ymax > 1.15:
                shift = ymax - 1.15
                for p in points_sorted:
                    p["y_adj"] -= shift
            if ymin < -1.15:
                shift = -1.15 - ymin
                for p in points_sorted:
                    p["y_adj"] += shift
            return points_sorted

        left = _spread([p for p in label_points if p["side"] == "left"])
        right = _spread([p for p in label_points if p["side"] == "right"])
        for p in left + right:
            x_text = 1.12 if p["side"] == "right" else -1.12
            ha = "left" if p["side"] == "right" else "right"
            ax.annotate(
                p["label"],
                xy=(0.92 * p["x"], 0.92 * p["y"]),
                xytext=(x_text, p["y_adj"]),
                ha=ha,
                va="center",
                fontsize=13,
                arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "gray"},
            )

        ax.set_title(res["display_name"], fontsize=16)
        ax.axis("equal")

    for j in range(n_agents, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Summed Greeks Attribution Breakdown", fontsize=19, y=0.98)
    fig.subplots_adjust(wspace=0.85, top=0.84)
    plt.show()