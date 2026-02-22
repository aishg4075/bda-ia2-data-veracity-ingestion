# Kafka Local Setup (Single-node KRaft)

This project uses Kafka as the default streaming source for the Topic 17 demo.

## Start Kafka

```bash
cd docker
docker-compose -f docker-compose.kafka.yml up -d
```

## Verify broker health

```bash
docker ps | grep topic17-kafka
```

Optional topic listing:

```bash
docker exec -it topic17-kafka kafka-topics.sh --bootstrap-server localhost:9092 --list
```

## Produce sample events

```bash
cd ..
python scripts/kafka_producer.py --input-csv data/sample_click_fraud.csv --topic clickstream_events
```

## Debug consume

```bash
python scripts/kafka_consumer_debug.py --topic clickstream_events --max-messages 10
```

## Stop Kafka

```bash
cd docker
docker-compose -f docker-compose.kafka.yml down
```

## Troubleshooting

- If port `9092` is busy, stop the conflicting process or remap to another port.
- If Spark cannot read Kafka, ensure the Spark Kafka package is downloaded (internet required the first time).
- If Docker Desktop is not used, start Colima first: `colima start`.
