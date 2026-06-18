#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// =====================================================
// WIFI
// =====================================================
const char* ssid     = "Quang Vinh";
const char* password = "vinh010203";

// =====================================================
// MQTT BROKER
// =====================================================
const char* MQTT_BROKER    = "22f19246dfe745fd9bbc373e63c12f9f.s1.eu.hivemq.cloud";
const int   MQTT_PORT      = 8883;

const char* MQTT_USER      = "ngoquangvinh";
const char* MQTT_PASSWORD  = "Vinh4953";

const char* MQTT_TOPIC     = "airquality/data";
const char* MQTT_CLIENT_ID = "esp32-air-sensor";
// =====================================================
// OLED
// =====================================================
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  -1
);

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

// =====================================================
// MQTT CLIENT
// =====================================================
WiFiClientSecure espClient;
PubSubClient mqttClient(espClient);

// =====================================================
// EMA FILTER
// =====================================================
float emaFilter(float oldValue,
                float newValue,
                float alpha) {

  return oldValue +
         alpha * (newValue - oldValue);
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

    if (Serial2.read() != 0x42)
      continue;

    unsigned long timeout = millis();

    while (Serial2.available() < 31) {

      if (millis() - timeout > 100)
        return false;
    }

    if (Serial2.read() != 0x4D)
      continue;

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
// OLED HELPER
// =====================================================
void oledStatus(const char* line1,
                const char* line2 = "",
                const char* line3 = "") {

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);

  display.setCursor(0, 10);
  display.println(line1);

  display.setCursor(0, 26);
  display.println(line2);

  display.setCursor(0, 42);
  display.println(line3);

  display.display();
}

// =====================================================
// OLED
// =====================================================
void drawDisplay() {

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);

  // HEADER
  display.setCursor(20, 0);
  display.println("AIR MONITOR");

  // DIVIDER
  display.drawLine(0, 10, 127, 10, WHITE);

  // DHT22
  display.setCursor(0, 14);
  display.print("T:");
  display.print(temperature, 1);
  display.print("C");

  display.setCursor(68, 14);
  display.print("H:");
  display.print(humidity, 0);
  display.print("%");

  // MQ135
  display.setCursor(0, 26);
  display.print("MQ:");
  display.print((int)mqFiltered);

  // PMS7003
display.setCursor(0, 38);
display.print("PM2.5:");
display.print(pm25Filtered, 1);

 display.setCursor(0, 50);
 display.print("PM10 :");
 display.print(pm10Filtered, 1);

  // MQTT status — góc phải, KHÔNG đè lên PM data
  display.setCursor(70, 26);
  display.print(mqttClient.connected() ? "[MQTT OK]" : "[NO MQTT]");

  display.display();
}

// =====================================================
// WIFI CONNECT
// =====================================================
void connectWiFi() {

  oledStatus("Connecting WiFi...", ssid);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");

  int dots = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    // Cập nhật OLED từng 500ms để người dùng thấy tiến trình
    char dotStr[16] = "";
    for (int i = 0; i < (dots % 4); i++) strcat(dotStr, ".");
    oledStatus("Connecting WiFi", ssid, dotStr);
    dots++;
  }

  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());

  char ipStr[24];
  WiFi.localIP().toString().toCharArray(ipStr, sizeof(ipStr));
  oledStatus("WiFi Connected!", ipStr);
  delay(1000);

  // GREEN
  setColor(LOW, HIGH, LOW);
}

// =====================================================
// MQTT CONNECT / RECONNECT
// =====================================================
void connectMQTT() {

  oledStatus("Connecting MQTT...", MQTT_BROKER);

  int attempt = 0;
  while (!mqttClient.connected()) {

    attempt++;
    char attemptStr[24];
    snprintf(attemptStr, sizeof(attemptStr), "Attempt #%d", attempt);
    oledStatus("Connecting MQTT...", MQTT_BROKER, attemptStr);

    Serial.print("Connecting MQTT...");

    if (mqttClient.connect(
        MQTT_CLIENT_ID,
        MQTT_USER,
        MQTT_PASSWORD)) {

      Serial.println("MQTT connected!");
      oledStatus("MQTT Connected!", MQTT_BROKER, MQTT_TOPIC);
      delay(800);

      // BLUE
      setColor(LOW, LOW, HIGH);

    } else {

      Serial.print("MQTT failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" retry in 3s");

      // RED
      setColor(HIGH, LOW, LOW);

      delay(3000);
    }
  }
}

// =====================================================
// PUBLISH MQTT
// =====================================================
void publishMQTT() {

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    return;
  }

  if (!mqttClient.connected()) {
    connectMQTT();
  }

  // Build JSON
  char payload[128];

  snprintf(payload, sizeof(payload),
    "{\"pm25\":%.1f,\"pm10\":%.1f,\"temp\":%.1f,\"hum\":%.1f,\"mq\":%.1f}",
    pm25Filtered,
    pm10Filtered,
    temperature,
    humidity,
    mqFiltered
  );

  bool ok = mqttClient.publish(MQTT_TOPIC, payload);

  Serial.println("==========");
  Serial.println(payload);
  Serial.print("MQTT publish: ");
  Serial.println(ok ? "OK" : "FAILED");

  if (ok) {
    // BLUE
    setColor(LOW, LOW, HIGH);
    tone(BUZZER_PIN, 2000, 100);
  } else {
    // RED
    setColor(HIGH, LOW, LOW);
    tone(BUZZER_PIN, 1000, 300);
  }
}

// =====================================================
// SETUP
// =====================================================
void setup() {

  Serial.begin(115200);

  // PMS7003 UART
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);

  dht.begin();

  pinMode(MQ135_PIN, INPUT);

  pinMode(LED_R, OUTPUT);
  pinMode(LED_G, OUTPUT);
  pinMode(LED_B, OUTPUT);

  pinMode(BUZZER_PIN, OUTPUT);

  // OLED — khởi tạo và hiện splash ngay
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED init FAILED!");
    while (1);
  }

  display.clearDisplay();
  display.display();
  oledStatus("AIR QUALITY AI", "Initializing...", "v1.0");
  delay(1000);

  // WIFI
  connectWiFi();

  // MQTT
  espClient.setInsecure();

  mqttClient.setServer(
    MQTT_BROKER,
    MQTT_PORT
  );
  connectMQTT();

  // PMS7003 WARMUP
  delay(5000);

  Serial.println("SYSTEM READY");
}

// =====================================================
// LOOP
// =====================================================
void loop() {

  // Giữ kết nối MQTT sống
  if (!mqttClient.connected()) {
    connectMQTT();
  }
  mqttClient.loop();

  unsigned long currentMillis = millis();

  // =========================================
  // READ SENSOR
  // =========================================
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
    Serial.print(" | PM10: "); Serial.print(pm10Filtered);
    Serial.print(" | MQ: "); Serial.println(mqFiltered);
  }

  // =========================================
  // PUBLISH MQTT
  // =========================================
  if (currentMillis - lastSend >= SEND_INTERVAL) {

    lastSend = currentMillis;

    publishMQTT();
  }
}
