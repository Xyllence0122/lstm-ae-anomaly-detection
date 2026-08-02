# V5 MQTT 端對端測試

## 1. 測試目的與研究邊界

本測試驗證以下完整資料路徑：

```text
PC 合成 sensor -> MQTT Broker -> Raspberry Pi 5
-> V5 schema/timestamp/window/runtime -> MQTT decision/alarm -> PC
```

PC 在同一程序內記錄送出與收到結果的 monotonic clock，因此端對端
round-trip latency 不受 PC 與 Pi 系統時鐘未同步影響。Pi 另在每筆結果中
回報 edge processing latency。兩者相減得到 MQTT 雙向傳輸、broker、queue
及 callback 的合計 overhead；此方法不能拆出單向網路延遲。

這是系統傳輸與 runtime 實驗。合成資料來自既有 generator family，不是新的
獨立 holdout，不得用本報告重新調整模型權重、profiles 或 threshold，也不能把
其中的 detection metrics 當成新的模型準確率證據。

## 2. 訊息安全契約

- MQTT 使用 QoS 1，不使用 retained sensor message。
- 每筆資料攜帶唯一 message ID、run/stream ID、sample index 與 sequence length。
- sensor schema 同時檢查 columns 順序與 manifest schema hash。
- 缺欄、多欄、錯序、重欄、NaN、Inf、跳號及 stream 中途改 context 均拒絕。
- 相同 message ID/sample index 的 QoS 1 重送只重發既有結果，不重複推論。
- ground-truth anomaly label 不傳給 Pi，避免標籤進入模型管線。

## 3. 安裝

PC 與 Pi 都在 repository 的 V5 branch 執行：

```bash
python -m pip install -r requirements-mqtt.txt
```

在 Pi 安裝 Mosquitto：

```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo mosquitto_passwd -c /etc/mosquitto/passwd tanet
sudo chown root:mosquitto /etc/mosquitto/passwd
sudo chmod 640 /etc/mosquitto/passwd
```

建立 `/etc/mosquitto/conf.d/tanet-v5.conf`：

```conf
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
```

重新啟動並確認：

```bash
sudo systemctl restart mosquitto
sudo systemctl --no-pager --full status mosquitto
hostname -I
```

此設定只應用於受控實驗室 LAN，不可在路由器開放 1883 到 Internet。跨不受信任
網路時應配置 TLS，兩支程式均支援 `--tls-ca`。

## 4. 先做同機 smoke test

Pi 終端機 1 啟動 edge service：

```bash
export MQTT_PASSWORD='設定的密碼'
python 34_run_v5_mqtt_edge.py \
  --broker 127.0.0.1 \
  --username tanet \
  --output outputs/v5/mqtt_edge_smoke_v5.json
```

Pi 終端機 2 啟動少量 sensor replay：

```bash
export MQTT_PASSWORD='設定的密碼'
python 35_run_v5_mqtt_sensor.py \
  --broker 127.0.0.1 \
  --username tanet \
  --normal 2 --type-a 1 --type-b 1 --type-c 1 \
  --publish-interval-ms 20 \
  --output outputs/v5/mqtt_e2e_smoke_v5.json
```

sensor 程式輸出報告後，回到終端機 1 按 `Ctrl+C`，讓 edge service 寫出報告。
同機 smoke test 只證明 MQTT 軟體管線，不包含 LAN 傳輸。

## 5. 正式 PC 到 Pi LAN 測試

Pi edge service 仍連到本機 broker：

```bash
export MQTT_PASSWORD='設定的密碼'
python 34_run_v5_mqtt_edge.py \
  --broker 127.0.0.1 \
  --username tanet \
  --output outputs/v5/mqtt_edge_formal_v5.json
```

Windows PC 的 VS Code PowerShell，將 `192.168.1.50` 換成 `hostname -I`
顯示的 Pi LAN IP：

```powershell
$env:MQTT_PASSWORD = "設定的密碼"
.\.venv\Scripts\python.exe 35_run_v5_mqtt_sensor.py `
  --broker 192.168.1.50 `
  --username tanet `
  --normal 100 --type-a 30 --type-b 30 --type-c 30 `
  --publish-interval-ms 20 `
  --output outputs/v5/mqtt_e2e_formal_v5.json
```

此組約產生兩萬筆訊息，主要測試 LAN/MQTT capacity 與 latency。若要補一組接近
真實 1 Hz cadence 的測試，可執行：

```powershell
.\.venv\Scripts\python.exe 35_run_v5_mqtt_sensor.py `
  --broker 192.168.1.50 `
  --username tanet `
  --normal 10 --type-a 0 --type-b 0 --type-c 0 `
  --publish-interval-ms 1019.9 `
  --output outputs/v5/mqtt_e2e_realtime_v5.json
```

不要同時啟動兩個 sensor simulator；若必須並行，需使用不同 `--topic-prefix`
及獨立 edge service。

## 6. 報告判讀

正式結果至少應符合：

- `status = complete`
- `missing_results = 0`
- `rejected_results = 0`
- `incomplete_sequences = 0`
- edge service 的 `queue_overflow_messages = 0`
- `decision_round_trip.p99_ms` 小於 nominal sensor interval `1019.9 ms`
- `external_alarm_round_trip` 有資料時，需報告 p50/p95/p99/max
- `offline_edge_decision_parity.passed = true` 且所有 decision mismatch 為 0
- detection metrics 只統計完整收到所有 sample 的序列

`mqtt_e2e_*_latencies.npz` 保存每筆 raw latency；JSON 內記錄該檔 SHA-256、
manifest hash、model version、statistics hash 與程式 hash。正式 JSON 與 NPZ 應一併
提交 Git，edge service 報告也應保留。
