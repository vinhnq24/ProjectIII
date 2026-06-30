#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <TinyGPS++.h>
#include "secrets.h"

// =====================================================
// WIFI
// =====================================================
const char* ssid     = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// =====================================================
// MQTT BROKER
// =====================================================
const char* MQTT_BROKER    = MQTT_BROKER_HOST;
const int   MQTT_PORT      = MQTT_BROKER_PORT;

const char* MQTT_USER      = MQTT_BROKER_USER;
const char* MQTT_PASSWORD  = MQTT_BROKER_PASSWORD;

const char* MQTT_TOPIC     = "airquality/data";
const char* MQTT_CLIENT_ID = "esp32-air-sensor";

// =====================================================
// OLED
// =====================================================
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// =====================================================
// DHT22
// =====================================================
#define DHTPIN  4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// =====================================================
// MQ135
// =====================================================
#define MQ135_PIN 34

// =====================================================
// PMS7003
// =====================================================
#define RXD2 16
#define TXD2 17

// =====================================================
// NEO-6M GPS 
// =====================================================
#define RXD1 32  
#define TXD1 33  

HardwareSerial GPS(1); 
TinyGPSPlus gps;

// =====================================================
// RGB LED (COMMON CATHODE)
// =====================================================
#define LED_R 25
#define LED_G 26
#define LED_B 14

// =====================================================
// BUZZER
// =====================================================
#define BUZZER_PIN 27

// =====================================================
// VARIABLES
// =====================================================
float temperature = 0;
float humidity    = 0;
float mqFiltered  = 0;

uint16_t pm1_0 = 0;
uint16_t pm2_5 = 0;
uint16_t pm10  = 0;

float pm25Filtered = 0;
float pm10Filtered = 0;

// Biến GPS
double gpsLat = 0.0;
double gpsLng = 0.0;
bool isGpsFixed = false;
int satellitesCount = 0;

// =====================================================
// FILTER
// =====================================================
const float ALPHA_PM = 0.25;
const float ALPHA_MQ = 0.2;

// =====================================================
// TIMING
// =====================================================
unsigned long lastSensorRead = 0;
unsigned long lastSend       = 0;

const unsigned long SENSOR_INTERVAL = 2000;
const unsigned long SEND_INTERVAL   = 5000;

WiFiClientSecure espClient;
PubSubClient mqttClient(espClient);

// =====================================================
// EMA FILTER
// =====================================================
float emaFilter(float oldValue, float newValue, float alpha) {
  return oldValue + alpha * (newValue - oldValue);
}

// =====================================================
// RGB
// =====================================================
void setColor(bool r, bool g, bool b) {
  digitalWrite(LED_R, r);
  digitalWrite(LED_G, g);
  digitalWrite(LED_B, b);
}

// =====================================================
// PMS7003
// =====================================================
bool readPMS7003() {
  static uint8_t buffer[32];
  while (Serial2.available() > 0) {
    if (Serial2.read() != 0x42) continue;
    unsigned long timeout = millis();
    while (Serial2.available() < 31) {
      if (millis() - timeout > 100) return false;
    }
    if (Serial2.read() != 0x4D) continue;

    buffer[0] = 0x42;
    buffer[1] = 0x4D;
    size_t len = Serial2.readBytes(&buffer[2], 30);
    if (len != 30) continue;

    uint16_t checksum = 0;
    for (int i = 0; i < 30; i++) checksum += buffer[i];
    uint16_t frameChecksum = (buffer[30] << 8) | buffer[31];
    if (checksum != frameChecksum) continue;

    pm1_0 = (buffer[10] << 8) | buffer[11];
    pm2_5 = (buffer[12] << 8) | buffer[13];
    pm10  = (buffer[14] << 8) | buffer[15];

    if (pm25Filtered == 0) {
      pm25Filtered = pm2_5;
      pm10Filtered = pm10;
    } else {
      pm25Filtered = emaFilter(pm25Filtered, pm2_5, ALPHA_PM);
      pm10Filtered = emaFilter(pm10Filtered, pm10,  ALPHA_PM);
    }
    return true;
  }
  return false;
}

// =====================================================
// MQ135
// =====================================================
void readMQ135() {
  long sum = 0;
  for (int i = 0; i < 10; i++) sum += analogRead(MQ135_PIN);
  float raw = sum / 10.0;
  if (mqFiltered == 0) {
    mqFiltered = raw;
  } else {
    mqFiltered = emaFilter(mqFiltered, raw, ALPHA_MQ);
  }
}

// =====================================================
// NEO-6M GPS READ (CẢI TIẾN LỌC SAI SỐ BẰNG HDOP)
// =====================================================
void readGPS() {
  while (GPS.available()) {
    gps.encode(GPS.read());
    yield(); 
  }

  satellitesCount = gps.satellites.value();
  
  // KIỂM TRA CHẶT CHẼ: Có tọa độ + Có dữ liệu HDOP + HDOP < 2.0 (Sai số phương ngang thấp) + Ít nhất 4 vệ tinh
  if (gps.location.isValid() && 
      gps.hdop.isValid() && 
      gps.hdop.value() < 500 && 
      satellitesCount >= 4) {
      
    isGpsFixed = true;
    gpsLat = gps.location.lat();
    gpsLng = gps.location.lng();
  } else {
    // Nếu rớt các điều kiện trên, ta coi như mất Fix để bảo vệ tính chính xác của dữ liệu
    isGpsFixed = false; 
  }
}

// =====================================================
// OLED HELPERS
// =====================================================
void oledStatus(const char* line1, const char* line2 = "", const char* line3 = "") {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 10);  display.println(line1);
  display.setCursor(0, 26);  display.println(line2);
  display.setCursor(0, 42);  display.println(line3);
  display.display();
}

void drawDisplay() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);

  // HEADER
  display.setCursor(20, 0);
  display.println("AIR MONITOR GPS");

  // DIVIDER
  display.drawLine(0, 9, 127, 9, WHITE);

  // DHT22
  display.setCursor(0, 12);
  display.print("T:"); display.print(temperature, 1); display.print("C");
  display.setCursor(55, 12);
  display.print("H:"); display.print(humidity, 0); display.print("%");

  // MQ135 & MQTT Status
  display.setCursor(0, 22);
  display.print("MQ:"); display.print((int)mqFiltered);
  display.setCursor(70, 22);
  display.print(mqttClient.connected() ? "[OK]" : "[ERR]");

  // PMS7003
  display.setCursor(0, 32); display.print("PM2.5:"); display.print(pm25Filtered, 1);
  display.setCursor(0, 42); display.print("PM10 :"); display.print(pm10Filtered, 1);

  // GPS DISPLAY 
  display.setCursor(0, 54);
  if (isGpsFixed) {
    display.print("GPS: ");
    display.print(gpsLat, 2);
    display.print(",");
    display.print(gpsLng, 2);
    display.print(" ("); display.print(satellitesCount); display.print("S)");
  } else {
    display.print("GPS: Wait Fix (");
    display.print(satellitesCount);
    display.print(")");
  }

  display.display();
}

// =====================================================
// WIFI & MQTT CONNECT
// =====================================================
void connectWiFi() {
  oledStatus("Connecting WiFi...", ssid);
  WiFi.begin(ssid, password);
  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    char dotStr[16] = "";
    for (int i = 0; i < (dots % 4); i++) strcat(dotStr, ".");
    oledStatus("Connecting WiFi", ssid, dotStr);
    dots++;
  }
  char ipStr[24];
  WiFi.localIP().toString().toCharArray(ipStr, sizeof(ipStr));
  oledStatus("WiFi Connected!", ipStr);
  delay(1000);
  setColor(LOW, HIGH, LOW);
}

void connectMQTT() {
  oledStatus("Connecting MQTT...", MQTT_BROKER);
  int attempt = 0;
  while (!mqttClient.connected()) {
    attempt++;
    char attemptStr[24];
    snprintf(attemptStr, sizeof(attemptStr), "Attempt #%d", attempt);
    oledStatus("Connecting MQTT...", MQTT_BROKER, attemptStr);

    if (mqttClient.connect(MQTT_CLIENT_ID, MQTT_USER, MQTT_PASSWORD)) {
      oledStatus("MQTT Connected!", MQTT_BROKER, MQTT_TOPIC);
      delay(500);
      setColor(LOW, LOW, HIGH);
    } else {
      setColor(HIGH, LOW, LOW);
      delay(3000);
    }
  }
}

// =====================================================
// PUBLISH MQTT
// =====================================================
void publishMQTT() {
  if (WiFi.status() != WL_CONNECTED) { connectWiFi(); return; }
  if (!mqttClient.connected()) { connectMQTT(); }

  char payload[224];
  // Thêm "ts" (timestamp dùng millis) để theo dõi khoảng thời gian gửi log dữ liệu
  snprintf(payload, sizeof(payload),
    "{\"ts\":%lu,\"pm25\":%.1f,\"pm10\":%.1f,\"temp\":%.1f,\"hum\":%.1f,\"mq\":%.1f,\"gps_fix\":%d,\"lat\":%.6f,\"lng\":%.6f}",
    millis(), pm25Filtered, pm10Filtered, temperature, humidity, mqFiltered, (isGpsFixed ? 1 : 0), gpsLat, gpsLng
  );

  bool ok = mqttClient.publish(MQTT_TOPIC, payload);
  Serial.println("==========");
  Serial.println(payload);
  Serial.print("MQTT publish: "); Serial.println(ok ? "OK" : "FAILED");

  if (ok) {
    setColor(LOW, LOW, HIGH);
    tone(BUZZER_PIN, 2000, 100);
  } else {
    setColor(HIGH, LOW, LOW);
    tone(BUZZER_PIN, 1000, 300);
  }
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);

  // PMS7003 UART (Serial2)
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);

  // NEO-6M GPS UART (HardwareSerial)
  GPS.begin(9600, SERIAL_8N1, RXD1, TXD1);

  dht.begin();
  pinMode(MQ135_PIN, INPUT);
  pinMode(LED_R, OUTPUT); pinMode(LED_G, OUTPUT); pinMode(LED_B, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED init FAILED!");
    while (1);
  }

  display.clearDisplay(); display.display();
  oledStatus("AIR QUALITY AI", "Initializing...", "v1.3 (Optimized)");
  delay(1000);

  connectWiFi();
  espClient.setInsecure();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  connectMQTT();

  delay(5000);
  Serial.println("SYSTEM READY");
}

// =====================================================
// LOOP
// =====================================================
void loop() {

  if (!mqttClient.connected()) { connectMQTT(); }
  mqttClient.loop();

  // Đọc GPS liên tục ở đầu loop
  readGPS();

  unsigned long currentMillis = millis();

  // READ SENSORS & UPDATE OLED
  if (currentMillis - lastSensorRead >= SENSOR_INTERVAL) {
    lastSensorRead = currentMillis;

    readPMS7003();

    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) {
      temperature = t;
      humidity    = h;
    }

    readMQ135();
    drawDisplay();

    Serial.print("PM2.5: "); Serial.print(pm25Filtered);
    Serial.print(" | MQ: "); Serial.print(mqFiltered);
    Serial.print(" | Satellites: "); Serial.println(satellitesCount);
    Serial.print(" | HDOP: "); Serial.println(gps.hdop.hdop());
  }

  // PUBLISH MQTT
  if (currentMillis - lastSend >= SEND_INTERVAL) {
    lastSend = currentMillis;
    publishMQTT();
  }
}
