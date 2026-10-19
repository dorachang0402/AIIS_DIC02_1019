/**
 * 50 Startups: Multiple Linear Regression & 3 Feature Selection Algorithms Explorer
 * Interactive Application Controller
 * Color Palette:
 * - Dark Slate Grey: #335c67
 * - Vanilla Custard: #fff3b0
 * - Honey Bronze:    #e09f3e
 * - Brown Red:       #9e2a2b
 * - Night Bordeaux:  #540b0e
 */

document.addEventListener("DOMContentLoaded", () => {
  // Brand Palette Constants
  const PALETTE = {
    darkSlateGrey: "#335c67",
    vanillaCustard: "#fff3b0",
    honeyBronze: "#e09f3e",
    brownRed: "#9e2a2b",
    nightBordeaux: "#540b0e",
    subtleBorder: "#e8e2d8",
    optimalGreen: "#2d6a4f",
    optimalBg: "#d8f3dc"
  };

  // State Management
  let currentAlgo = "Forward SFS";
  let currentK = 2;

  // Chart Instances
  let rmseChartInstance = null;
  let r2ChartInstance = null;
  let multiAlgoChartInstance = null;

  // DOM Elements
  const algoTabs = document.querySelectorAll(".algo-tab-btn");
  const kButtons = document.querySelectorAll(".k-btn");

  const valRMSE = document.getElementById("valRMSE");
  const valR2 = document.getElementById("valR2");
  const valAdjR2 = document.getElementById("valAdjR2");
  const valMAE = document.getElementById("valMAE");
  const valFStat = document.getElementById("valFStat");
  const badgeRMSE = document.getElementById("badgeRMSE");
  const parsimonyBox = document.getElementById("parsimonyBox");
  const parsimonyTitle = document.getElementById("parsimonyTitle");
  const parsimonyDesc = document.getElementById("parsimonyDesc");

  const chart1Badge = document.getElementById("chart1Badge");
  const chart2Badge = document.getElementById("chart2Badge");
  const equationText = document.getElementById("equationText");
  const coefTableBody = document.getElementById("coefTableBody");
  const outliersTableBody = document.getElementById("outliersTableBody");
  const featuresCountBadge = document.getElementById("featuresCountBadge");

  // Simulator Elements
  const sliderRD = document.getElementById("sliderRD");
  const sliderMkt = document.getElementById("sliderMkt");
  const sliderAdmin = document.getElementById("sliderAdmin");
  const selectState = document.getElementById("selectState");

  const dispRD = document.getElementById("dispRD");
  const dispMkt = document.getElementById("dispMkt");
  const dispAdmin = document.getElementById("dispAdmin");
  const dispState = document.getElementById("dispState");

  const groupRD = document.getElementById("groupRD");
  const groupMkt = document.getElementById("groupMkt");
  const groupAdmin = document.getElementById("groupAdmin");
  const groupState = document.getElementById("groupState");

  const predProfitVal = document.getElementById("predProfitVal");
  const bdIntercept = document.getElementById("bdIntercept");
  const bdRD = document.getElementById("bdRD");
  const bdMkt = document.getElementById("bdMkt");
  const bdAdmin = document.getElementById("bdAdmin");
  const bdState = document.getElementById("bdState");

  // Presets
  const presetTech = document.getElementById("presetTech");
  const presetGrowth = document.getElementById("presetGrowth");
  const presetBootstrap = document.getElementById("presetBootstrap");

  // Helpers
  const formatCurrency = (val) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(val);
  };

  const formatNumber = (val, decimals = 4) => {
    return Number(val).toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  };

  // Get Current Model Object
  const getActiveModel = () => {
    const list = window.MODEL_DATA[currentAlgo] || window.MODEL_DATA["Forward SFS"];
    return list.find((item) => item.k === currentK) || list[1];
  };

  // 1. Initialize Dual Charts
  const initCharts = () => {
    const ctxRMSE = document.getElementById("rmseChart").getContext("2d");
    const ctxR2 = document.getElementById("r2Chart").getContext("2d");

    const steps = window.MODEL_DATA[currentAlgo];
    const labels = steps.map((s) => `k=${s.k}`);
    const rmseData = steps.map((s) => s.rmse);
    const r2Data = steps.map((s) => s.r2);
    const adjR2Data = steps.map((s) => s.adj_r2);

    // RMSE Chart
    rmseChartInstance = new Chart(ctxRMSE, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "RMSE (越低越好)",
            data: rmseData,
            borderColor: PALETTE.darkSlateGrey,
            backgroundColor: "rgba(51, 92, 103, 0.12)",
            borderWidth: 3,
            fill: true,
            tension: 0.25,
            pointBackgroundColor: steps.map((s) =>
              s.k === currentK ? PALETTE.honeyBronze : "#ffffff"
            ),
            pointBorderColor: steps.map((s) =>
              s.k === currentK ? PALETTE.nightBordeaux : PALETTE.darkSlateGrey
            ),
            pointBorderWidth: 2.5,
            pointRadius: steps.map((s) => (s.k === currentK ? 8 : 5)),
            pointHoverRadius: 9
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: PALETTE.nightBordeaux,
            titleColor: PALETTE.vanillaCustard,
            bodyColor: "#ffffff",
            padding: 10,
            callbacks: {
              label: (ctx) => `RMSE: $${Number(ctx.raw).toFixed(2)}`
            }
          }
        },
        scales: {
          y: {
            grid: { color: "rgba(0,0,0,0.05)" },
            ticks: {
              font: { family: "Inter", size: 11 },
              callback: (v) => `$${v.toLocaleString()}`
            }
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", weight: "600" } }
          }
        }
      }
    });

    // R2 Chart
    r2ChartInstance = new Chart(ctxR2, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "R-squared (解釋力)",
            data: r2Data,
            borderColor: PALETTE.brownRed,
            backgroundColor: "rgba(158, 42, 43, 0.08)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.25,
            pointBackgroundColor: steps.map((s) =>
              s.k === currentK ? PALETTE.honeyBronze : "#ffffff"
            ),
            pointBorderColor: steps.map((s) =>
              s.k === currentK ? PALETTE.nightBordeaux : PALETTE.brownRed
            ),
            pointBorderWidth: 2.5,
            pointRadius: steps.map((s) => (s.k === currentK ? 8 : 5))
          },
          {
            label: "Adjusted R²",
            data: adjR2Data,
            borderColor: PALETTE.honeyBronze,
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 3,
            fill: false,
            tension: 0.25
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { font: { family: "Inter", size: 11, weight: "600" } }
          },
          tooltip: {
            backgroundColor: PALETTE.darkSlateGrey,
            titleColor: PALETTE.vanillaCustard,
            bodyColor: "#ffffff",
            padding: 10,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${(Number(ctx.raw) * 100).toFixed(2)}%`
            }
          }
        },
        scales: {
          y: {
            grid: { color: "rgba(0,0,0,0.05)" },
            ticks: {
              font: { family: "Inter", size: 11 },
              callback: (v) => `${(v * 100).toFixed(1)}%`
            }
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", weight: "600" } }
          }
        }
      }
    });
  };

  // 2. Initialize 3-Algorithm Multi-Comparison Chart
  const initMultiAlgoChart = () => {
    const ctx = document.getElementById("multiAlgoChart").getContext("2d");
    const labels = ["k = 1", "k = 2", "k = 3", "k = 4", "k = 5"];

    const forwardRMSE = window.MODEL_DATA["Forward SFS"].map((s) => s.rmse);
    const rfeRMSE = window.MODEL_DATA["RFE"].map((s) => s.rmse);
    const lassoRMSE = window.MODEL_DATA["Lasso"].map((s) => s.rmse);

    multiAlgoChartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Forward SFS (向前逐步)",
            data: forwardRMSE,
            borderColor: PALETTE.darkSlateGrey,
            backgroundColor: PALETTE.darkSlateGrey,
            borderWidth: 2.8,
            pointRadius: 6,
            pointHoverRadius: 8,
            tension: 0.2
          },
          {
            label: "RFE (遞迴消除)",
            data: rfeRMSE,
            borderColor: PALETTE.brownRed,
            backgroundColor: PALETTE.brownRed,
            borderWidth: 2.8,
            pointRadius: 6,
            pointHoverRadius: 8,
            tension: 0.2
          },
          {
            label: "Lasso L1 (正則稀疏路徑)",
            data: lassoRMSE,
            borderColor: PALETTE.honeyBronze,
            backgroundColor: PALETTE.honeyBronze,
            borderWidth: 3,
            borderDash: [6, 4],
            pointRadius: 6,
            pointHoverRadius: 8,
            tension: 0.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: {
              font: { family: "Inter", size: 12, weight: "700" },
              boxWidth: 16,
              usePointStyle: true
            }
          },
          tooltip: {
            backgroundColor: PALETTE.nightBordeaux,
            titleColor: PALETTE.vanillaCustard,
            padding: 12,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: RMSE $${Number(ctx.raw).toFixed(2)}`
            }
          }
        },
        scales: {
          y: {
            title: {
              display: true,
              text: "RMSE (越低代表預測越精確)",
              font: { family: "Inter", weight: "600", size: 12 }
            },
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              callback: (v) => `$${v.toLocaleString()}`
            }
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: "Outfit", weight: "700", size: 12 } }
          }
        }
      }
    });
  };

  // 3. Update Visuals & DOM on State Change
  const updateUI = () => {
    const model = getActiveModel();
    const steps = window.MODEL_DATA[currentAlgo] || window.MODEL_DATA["Forward SFS"];

    // Update KPI metrics
    valRMSE.textContent = formatCurrency(model.rmse);
    valR2.textContent = formatNumber(model.r2, 4);
    valAdjR2.textContent = formatNumber(model.adj_r2, 4);
    valMAE.textContent = formatCurrency(model.mae);
    valFStat.textContent = `${formatNumber(model.f_stat, 2)}`;

    // Badges & Parsimony
    if (currentK === 2) {
      badgeRMSE.textContent = "★ 全局最佳 (Min RMSE)";
      badgeRMSE.style.background = PALETTE.vanillaCustard;
      badgeRMSE.style.color = PALETTE.nightBordeaux;
      parsimonyTitle.textContent = `★ 最優模型鎖定：${currentAlgo} 在 k = 2 達成最佳帕西莫尼精簡平衡`;
      parsimonyDesc.innerHTML = `特徵組合僅需 <strong>[${model.features.join(", ")}]</strong>，在測試集獲得最低誤差（$${model.rmse.toFixed(
        2
      )}）與最高解釋力（${(model.r2 * 100).toFixed(
        2
      )}%），為 VC 評估與資本分配首選推薦！`;
    } else if (currentK === 1) {
      badgeRMSE.textContent = "單變數基準";
      badgeRMSE.style.background = "#e2e8f0";
      badgeRMSE.style.color = "#334155";
      parsimonyTitle.textContent = `單一關鍵驅動力：研發支出 (R&D Spend)`;
      parsimonyDesc.innerHTML = `僅依靠 R&D Spend 即可解釋超過 94.6% 的獲利變異，顯示創新研發為新創公司的核心支柱。`;
    } else {
      badgeRMSE.textContent = "過擬合警示 (Overfitting)";
      badgeRMSE.style.background = "#fee2e2";
      badgeRMSE.style.color = "#991b1b";
      parsimonyTitle.textContent = `邊際效益遞減警示：納入 ${currentK} 個特徵導致 RMSE 上升`;
      parsimonyDesc.innerHTML = `當增添行政支出或地區變數後，RMSE 反由 $8,198.8 攀升至 <strong>$${model.rmse.toFixed(
        2
      )}</strong>，證明多餘變數引入雜訊與共線性，損害模型泛化性。`;
    }

    // Update Chart Badges
    chart1Badge.textContent = `${currentAlgo} (k=${currentK})`;
    chart2Badge.textContent = `${currentAlgo} (k=${currentK})`;
    featuresCountBadge.textContent = `${model.features.length} 個特徵入選`;

    // Update Charts
    if (rmseChartInstance && r2ChartInstance) {
      const rmseData = steps.map((s) => s.rmse);
      const r2Data = steps.map((s) => s.r2);
      const adjR2Data = steps.map((s) => s.adj_r2);

      rmseChartInstance.data.datasets[0].data = rmseData;
      rmseChartInstance.data.datasets[0].pointBackgroundColor = steps.map((s) =>
        s.k === currentK ? PALETTE.honeyBronze : "#ffffff"
      );
      rmseChartInstance.data.datasets[0].pointRadius = steps.map((s) =>
        s.k === currentK ? 8 : 5
      );
      rmseChartInstance.update();

      r2ChartInstance.data.datasets[0].data = r2Data;
      r2ChartInstance.data.datasets[1].data = adjR2Data;
      r2ChartInstance.data.datasets[0].pointBackgroundColor = steps.map((s) =>
        s.k === currentK ? PALETTE.honeyBronze : "#ffffff"
      );
      r2ChartInstance.data.datasets[0].pointRadius = steps.map((s) =>
        s.k === currentK ? 8 : 5
      );
      r2ChartInstance.update();
    }

    // Update Equation Text
    const intercept = model.coefficients.find((c) => c.feature.includes("Intercept"));
    let eq = `Profit = ${formatNumber(intercept ? intercept.coef : 0, 2)}`;
    model.coefficients.forEach((c) => {
      if (!c.feature.includes("Intercept")) {
        const sign = c.coef >= 0 ? "+" : "-";
        eq += ` ${sign} ${formatNumber(Math.abs(c.coef), 4)} × (${c.feature})`;
      }
    });
    equationText.textContent = eq;

    // Update Coefficients Table
    coefTableBody.innerHTML = "";
    model.coefficients.forEach((c) => {
      const tr = document.createElement("tr");
      const isSig = c.pvalue !== null && c.pvalue < 0.05;
      const sigBadge = isSig
        ? `<span class="badge-sig">顯著 (p < 0.05)</span>`
        : `<span class="badge-nonsig">不顯著</span>`;

      const ciText =
        c.ci_lower !== null && c.ci_upper !== null
          ? `[${c.ci_lower.toFixed(2)}, ${c.ci_upper.toFixed(2)}]`
          : "-";
      const vifText = c.vif !== null ? c.vif.toFixed(2) : "-";

      tr.innerHTML = `
        <td><strong>${c.feature}</strong></td>
        <td class="num-cell"><strong>${c.coef.toFixed(4)}</strong></td>
        <td class="num-cell">${c.stderr !== null ? c.stderr.toFixed(4) : "-"}</td>
        <td class="num-cell">${c.tstat !== null ? c.tstat.toFixed(2) : "-"}</td>
        <td class="num-cell">${c.pvalue !== null ? (c.pvalue < 0.0001 ? "< 0.0001" : c.pvalue.toFixed(4)) : "-"} ${sigBadge}</td>
        <td class="num-cell">${ciText}</td>
        <td class="num-cell">${vifText}</td>
      `;
      coefTableBody.appendChild(tr);
    });

    // Update Residuals / Outliers Table
    outliersTableBody.innerHTML = "";
    if (model.outliers) {
      model.outliers.forEach((o) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>#${o.rank}</td>
          <td>Sample #${o.index}</td>
          <td class="num-cell">${formatCurrency(o.actual)}</td>
          <td class="num-cell">${formatCurrency(o.predicted)}</td>
          <td class="num-cell" style="color: ${o.residual >= 0 ? "#166534" : "#991b1b"}">
            ${o.residual >= 0 ? "+" : ""}${formatCurrency(o.residual)}
          </td>
          <td class="num-cell"><strong>${o.pct_error.toFixed(2)}%</strong></td>
        `;
        outliersTableBody.appendChild(tr);
      });
    }

    // Update Simulator input states based on whether features are in model
    const activeFeats = model.features;
    const hasAdmin = activeFeats.some((f) => f.includes("Administration"));
    const hasMkt = activeFeats.some((f) => f.includes("Marketing Spend"));
    const hasRD = activeFeats.some((f) => f.includes("R&D Spend"));
    const hasState = activeFeats.some((f) => f.includes("State"));

    groupAdmin.classList.toggle("disabled", !hasAdmin);
    groupMkt.classList.toggle("disabled", !hasMkt);
    groupRD.classList.toggle("disabled", !hasRD);
    groupState.classList.toggle("disabled", !hasState);

    // Recompute Simulator
    calculateSimulation();
  };

  // 4. Simulator Calculation
  const calculateSimulation = () => {
    const model = getActiveModel();
    const rdVal = parseFloat(sliderRD.value);
    const mktVal = parseFloat(sliderMkt.value);
    const adminVal = parseFloat(sliderAdmin.value);
    const stateVal = selectState.value;

    dispRD.textContent = `$${rdVal.toLocaleString()}`;
    dispMkt.textContent = `$${mktVal.toLocaleString()}`;
    dispAdmin.textContent = `$${adminVal.toLocaleString()}`;
    dispState.textContent = `${stateVal} ${stateVal === "California" ? "(基準組)" : ""}`;

    // Get model coefficients
    const interceptObj = model.coefficients.find((c) => c.feature.includes("Intercept"));
    const coefRD = model.coefficients.find((c) => c.feature === "R&D Spend");
    const coefMkt = model.coefficients.find((c) => c.feature === "Marketing Spend");
    const coefAdmin = model.coefficients.find((c) => c.feature === "Administration");
    const coefNY = model.coefficients.find((c) => c.feature === "State_New York");
    const coefFL = model.coefficients.find((c) => c.feature === "State_Florida");

    let totalProfit = interceptObj ? interceptObj.coef : 0;
    bdIntercept.textContent = formatCurrency(interceptObj ? interceptObj.coef : 0);

    // RD Contribution
    if (coefRD) {
      const cRD = rdVal * coefRD.coef;
      totalProfit += cRD;
      bdRD.textContent = `${cRD >= 0 ? "+" : ""}${formatCurrency(cRD)} (+${coefRD.coef.toFixed(2)}/$)`;
    } else {
      bdRD.textContent = "未納入此模型 ($0.00)";
    }

    // Marketing Contribution
    if (coefMkt) {
      const cMkt = mktVal * coefMkt.coef;
      totalProfit += cMkt;
      bdMkt.textContent = `${cMkt >= 0 ? "+" : ""}${formatCurrency(cMkt)} (+${coefMkt.coef.toFixed(2)}/$)`;
    } else {
      bdMkt.textContent = "未納入此模型 ($0.00)";
    }

    // Admin Contribution
    if (coefAdmin) {
      const cAdmin = adminVal * coefAdmin.coef;
      totalProfit += cAdmin;
      bdAdmin.textContent = `${cAdmin >= 0 ? "+" : ""}${formatCurrency(cAdmin)} (+${coefAdmin.coef.toFixed(2)}/$)`;
    } else {
      bdAdmin.textContent = "未納入或無顯著影響 ($0.00)";
    }

    // State Contribution
    let stateContrib = 0;
    let stateDesc = "加州基準組 ($0.00)";
    if (stateVal === "New York" && coefNY) {
      stateContrib = coefNY.coef;
      stateDesc = `紐約加乘 (+${formatCurrency(stateContrib)})`;
    } else if (stateVal === "Florida" && coefFL) {
      stateContrib = coefFL.coef;
      stateDesc = `佛州效應 (${formatCurrency(stateContrib)})`;
    }
    totalProfit += stateContrib;
    bdState.textContent = stateDesc;

    // Display Total Profit
    predProfitVal.textContent = formatCurrency(totalProfit);
  };

  // 5. Setup Event Listeners
  // Algorithm Tab Switching
  algoTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      algoTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const selected = tab.getAttribute("data-algo");
      if (selected === "Compare") {
        document.getElementById("chartsSection").scrollIntoView({ behavior: "smooth" });
        return;
      }

      currentAlgo = selected;
      updateUI();
    });
  });

  // Feature Count (k) Buttons
  kButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      kButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      currentK = parseInt(btn.getAttribute("data-k"), 10);
      updateUI();
    });
  });

  // Simulator Sliders & Select
  [sliderRD, sliderMkt, sliderAdmin].forEach((slider) => {
    slider.addEventListener("input", calculateSimulation);
  });
  selectState.addEventListener("change", calculateSimulation);

  // Quick Presets
  presetTech.addEventListener("click", () => {
    sliderRD.value = 165000;
    sliderMkt.value = 470000;
    sliderAdmin.value = 136000;
    selectState.value = "New York";
    calculateSimulation();
  });

  presetGrowth.addEventListener("click", () => {
    sliderRD.value = 85000;
    sliderMkt.value = 380000;
    sliderAdmin.value = 140000;
    selectState.value = "Florida";
    calculateSimulation();
  });

  presetBootstrap.addEventListener("click", () => {
    sliderRD.value = 35000;
    sliderMkt.value = 60000;
    sliderAdmin.value = 80000;
    selectState.value = "California";
    calculateSimulation();
  });

  // 6. Bootstrap Initial State
  initCharts();
  initMultiAlgoChart();
  updateUI();
});
