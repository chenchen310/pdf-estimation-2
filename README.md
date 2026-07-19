# psf_estimator — BBP defect impulse-response 萃取與 synthetic defect 生成

從 KLA Broadband Plasma (BBP) 檢測影像（defect / reference pair，12-bit）直接估計
defect 的 **effective impulse response**，並用它把 synthetic defect 貼回真實影像，
產生含 pixel-level ground truth 的 U-Net 訓練資料。

> **本 repo 另含一條 GDS → BBP 預測 pipeline**（`d2db/`，Stage 0）：從 GDS layer
> 光柵直接預測 BBP 影像，作為 die-to-database 參考，與上述缺陷萃取共用同一套
> 光學／對位／噪聲骨架。見下方〈GDS → BBP〉章節。

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

## GDS → BBP (die-to-database) 預測 pipeline

從 GDS layer 光柵預測 BBP 影像，讓預測結果可作為 die-to-database 檢測參考。
本站點最小特徵 < 20 nm，遠低於 ~100 nm 可解析極限，所有 pattern 完全
sub-resolution：影像只承載**局部平均的等效反射率**。據此 Stage-0 模型對
band-limited 的區域密度為**線性**：

```
Y ≈ g · Σ_k w_k · D_k(t, σx) + b
```

- 區域 = GDS layer 的 Boolean 組合 × 方向類別（iso／水平／垂直，由結構張量
  分類）。方向切分是因為 VN（垂直線偏振）照明使次波長密集光柵的等效反射率
  各向異性。
- `w_k`：每區域一個等效反射率（灰階尺度）；`g, b`：per-frame 增益／偏移；
  `t`：per-frame 設計↔影像對位；`σx`：一個共用的 Gaussian kernel 膨脹，吸收
  名義光學與真機的差異。

| Stage | 內容 | 模組 |
|---|---|---|
| — | Frame／layer 探索、光柵慣例正規化（bool / uint8 / float 皆可） | `d2db/io_utils.py` |
| — | 區域分解（Boolean combo × 方向）；罕見組合遮罩＋標記、不外插 | `d2db/regions.py` |
| — | 名義寬頻 OTF 渲染 + s×s pixel 積分（hi-res blur 快取於 `<run>/cache/`） | `d2db/render.py` |
| 0 | 交替校準：全域 LS 解 `w` ↔ per-frame robust gain/offset ↔ 對位 ↔ 共用 σx | `d2db/calibrate.py` |
| eval | 凍結模型、per-frame 只擬合 (g,b,t)；z = resid/σ_noise、nuisance vs τ | `d2db/evald2db.py` |

關鍵技術點：

- **對位用高通交叉相關（`_cc_shift`）而非 `measure_shift`（Lucas–Kanade）**：
  pattern 全 sub-resolution 時梯度能量集中在稀疏的巨觀邊界、落在 LK 的噪聲
  特徵值底檻以下而完全失效（實測現象）。CC + 拋物線峰值內插對 pattern 不變
  方向自然回報 ~0，行為同樣優雅。
- **罕見區域組合絕不外插**：面積低於 `min_region_area` 的 combo 直接丟棄、其
  像素遮罩並標記，不用無資料的權重去猜。
- **評估鏡像生產流程**：模型凍結，σ_noise 由每張 frame 自身的帶外頻譜
  （> 0.42 cyc/px 只剩感測器噪聲）逐像素量測；sim 資料額外分離 σ_model 與 σ_noise。

### 使用

```bash
# 1) 產生 GDS-like 模擬資料（FID_{i}_bbp.npy + FID_{i}_{OD,POLY}.npy 設計光柵）
.venv/bin/python scripts/make_gds_test_data.py --out data/gds_sim --n-layouts 16 --dies 4 --seed 0

# 2) Stage-0 校準（真實 fab 資料轉成同一契約後用法相同）
.venv/bin/python scripts/run_d2db.py --data-dir data/gds_sim --out-dir runs/gds_sim --split die

# 3) Held-out 評估（z-map、nuisance vs τ；有 truth/ 時額外算 σ_model/σ_noise）
.venv/bin/python scripts/evaluate_d2db.py --run-dir runs/gds_sim --data-dir data/gds_sim --set test
```

**資料契約**（真實資料須轉成此格式）：`FID_{i:05d}_bbp.npy`（uint16 256×256）
＋ 每層 `FID_{i:05d}_{LAYER}.npy`（2048×2048 面積覆蓋率光柵，pixel_nm/8 =
3.75 nm；uint8 / binary / float 皆可，loader 自動正規化並記錄慣例）。Layer 集合
由 config 驅動（`D2DBConfig.layer_names`，預設從檔名自動探索）——換站點帶其他
layer 時不需改碼。

### 現況與限制

Stage 0（線性密度模型）已在 64-frame 模擬替身上驗證：對位 ~0.07 px、median
Pearson r ≈ 0.99、NRMSE ≈ 4.5%，且 layout-split 僅略差於 die-split（對沒看過的
圖形幾乎不掉分，是物理模型相對純 ML 的核心優勢）。誠實的天花板：σ_model <
σ_noise 的面積僅約 35%，殘差集中在通帶內中頻（區塊灰階、邊緣、離焦相位效應）。
**Stage 1**（複數場反射率 + 部分同調 Abbe／SOCS 成像，pupil 吃 ECP 光源、focus、
VN 偏振）即針對此殘差，是下一步。模擬器 `sim/simulate_gds.py` 刻意比估計器豐富
（部分同調、複數反射率、per-die 膜漂移、CD bias／圓角、VN 各向異性代理），使
Stage-0 無法「作弊」。

## Repo 結構

```
psfest/    缺陷 impulse-response 估計與合成核心（config/optics/io/diffimg/noise/detect/epsf/validate/synth/report）
d2db/      GDS → BBP 預測核心（config/io_utils/regions/render/calibrate/evald2db）
sim/       模擬器：simulate.py（缺陷，6x 高解析光學鏈）、simulate_gds.py（GDS→BBP，部分同調 Abbe）
scripts/   缺陷：make_test_data / run_pipeline / evaluate_sim / run_synth ｜ GDS：make_gds_test_data / run_d2db / evaluate_d2db
data/      測試資料（生成；真實 fab 資料亦落於此，永不進 git）
runs/      pipeline 輸出
```
