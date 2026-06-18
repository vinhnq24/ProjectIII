import paho.mqtt.client as mqtt
import ssl

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TLS,
)

def on_message(client, userdata, msg):
    print("Received:", msg.payload.decode())

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

if MQTT_USERNAME:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

if MQTT_TLS:
    client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(False)

client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC)

print(f"Waiting for data on {MQTT_BROKER}:{MQTT_PORT} topic {MQTT_TOPIC}...")
client.loop_forever()
