# psf_estimator — BBP defect impulse-response 萃取與 synthetic defect 生成

從 KLA Broadband Plasma (BBP) 檢測影像（defect / reference pair，12-bit）直接估計
defect 的 **effective impulse response**，並用它把 synthetic defect 貼回真實影像，
產生含 pixel-level ground truth 的 U-Net 訓練資料。

## 模型與核心觀念

BBP 是部分同調成像：次解析度 defect 在 signed difference image 中的訊號主要來自
散射場與背景場的干涉交叉項，因此**振幅/極性隨局部背景變化、形狀（固定 mode 與
focus 下）近似不變**。據此把每個事件建模為

```
D_i(x) = a_i * h(x - c_i) + b_i + noise
```

- `h`：共同 impulse response（oversampled 網格、unit peak；一次吃進光學 PSF、
  cross-term、pixel 積分的總效應）
- `a_i`：帶符號振幅（暗 defect 為負）；`c_i`：sub-pixel 中心；`b_i`：局部殘餘背景

合成時以 *strength* = `-a / (I_local - pedestal)`（局部背景吸收比例）參數化振幅，
使振幅自動隨背景縮放，符合 cross-term 物理。

## Pipeline（六階段）

| Stage | 內容 | 模組 |
|---|---|---|
| 0 | 噪聲模型 `sigma_diff(I, |grad I|)`（MAD、對 defect 穩健；低梯度邊際曲線供合成） | `psfest/noise.py` |
| 1 | Robust gain/offset 匹配 + signed diff + 殘餘位移 QC（gradient-based 估計；>0.1px 自動 re-register） | `psfest/diffimg.py` |
| 2 | z-map 事件偵測（雙極性）＋點狀篩選 gate（集中度/二階矩/單一 blob/飽和/crowding），無標註 → gate 全記錄＋人工抽查 gallery | `psfest/detect.py` |
| 3 | ePSF 交替估計（Anderson–King 式累積、band-limit projection、Huber 權重、χ² 離群剔除） | `psfest/epsf.py` |
| 4 | 驗證：k-fold held-out 殘差、FWHM vs 理論、頻譜 band-limit 一致性、gallery | `psfest/validate.py` |
| 5 | Synthetic defect 生成（strength 取樣、Fourier sub-pixel 渲染、噪聲增量、12-bit clip、diff 重算、mask 標註） | `psfest/synth.py` |

關鍵技術點：

- **Band-limit projection 是物理正則化**：光學影像頻譜嚴格侷限於
  `2NA/λ_min`（本組參數 = 0.3 cycles/pixel），每次迭代把 `h` 投影回該支撐。
  驗證指標 `bandlimit_violation_raw` 回報「未投影的原始累積」有多少能量出界。
- **位移量測用 gradient-based（Lucas–Kanade 式）而非 phase correlation**：
  週期 pattern 下 phase correlation 有週期 alias 且沿 bar 方向無資訊；
  structure tensor 的特徵值噪聲底檻會把「不產生殘影的方向」正確回報為 0。
- **Fourier shift 是精確內插**（影像 band-limited 且取樣高於 Nyquist），
  對齊/渲染不引入模糊——naive 平均會把 h 抹寬、讓合成 defect 比真實更糊。

## 使用

```bash
# 環境（一次）
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy matplotlib

# 1) 產生模擬測試資料（DID_{index}_def.npy / DID_{index}_ref1.npy + truth/）
.venv/bin/python scripts/make_test_data.py --out data/sim --n 650 --seed 0

# 2) 跑 pipeline（真實資料同樣用法，指到資料夾即可）
.venv/bin/python scripts/run_pipeline.py --data-dir data/sim --out-dir runs/sim650 --kfold

# 3) 模擬資料限定：對 ground truth 打分（pipeline 全程看不到 truth/）
.venv/bin/python scripts/evaluate_sim.py --run-dir runs/sim650 --data-dir data/sim

# 4) 生成 U-Net 訓練資料（SYN_{i}_def/_ref1/_diff/_mask.npy + catalog.csv）
.venv/bin/python scripts/run_synth.py --run-dir runs/sim650 --out-dir data/synth --n 2000
```

換 optical mode 時：建立新的 config JSON（`PipelineConfig.to_json` 的格式，改
`optics` 的 pixel_nm / na / λ band），以 `--config` 傳入；所有視窗、截止頻率、
oversampling 相關量都由 optics 導出。

## Run 產出

```
runs/<name>/
  config.json          # 完整參數快照
  noise_model.npz      # sigma_diff(I, |grad|) + pedestal
  events.csv           # 全部偵測事件：位置、z、gate 特徵、擬合 (a,b,c,chi2,rho)
  psf.npz              # h（oversampled）、s、R、光學參數
  metrics.json         # FWHM、band-limit 檢查、k-fold、事件統計
  run_summary.json     # clean pair 清單、strength 分布擬合（供合成）
  figs/                # 噪聲曲線、QC、h 影像/剖面/MTF、event/reject gallery、
                       # amplitude vs background、（eval 後）與 GT 比較圖
```

## 已知近似與注意事項

1. **暗 defect 的噪聲**：合成時無法從真實影像「移除」shot noise，最深像素約有
   ≤15% 的局部過噪（二階效應，僅數個 pixel）。
2. **Pedestal 由資料百分位估計**：strength 的絕對尺度因此有系統偏差，但擬合與
   合成使用同一 pedestal，自洽抵銷，不影響合成資料品質。
3. **單一 h 假設**：`metrics.json` 的 `rho` 直方圖（`figs/chi2_rho.png`）用於
   檢查形狀家族；若雙峰（focus 漂移、多 mode 混入），應分群各估一組 h。
4. **形狀不變假設**：`amplitude_vs_bg.png` 檢查振幅-背景關係；若不同背景上形狀
   顯著不同（強 cross-term 相位效應），需按背景 context 分組估計。
5. **建議的合成品質驗收**：訓練小分類器區分真實/合成 patch（同 SNR 區間），
   AUC ≈ 0.5 為達標。
6. 篩選門檻（`tau_*`、集中度、RMS 半徑上限）預設值以模擬資料校過；換真實資料
   先看 `figs/reject_gallery.png` 與 `events.csv` 的 gate 特徵再微調。

## Repo 結構

```
psfest/    估計與合成核心（config/optics/io/diffimg/noise/detect/epsf/validate/synth/report）
sim/       模擬器（獨立的 6x 高解析光學鏈，供端到端驗證；估計器不共用其假設）
scripts/   make_test_data / run_pipeline / evaluate_sim / run_synth
data/      測試資料（生成）
runs/      pipeline 輸出
```
