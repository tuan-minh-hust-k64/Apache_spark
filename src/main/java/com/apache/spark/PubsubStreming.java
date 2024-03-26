package com.apache.spark;

import com.google.api.core.ApiFuture;
import com.google.cloud.bigquery.storage.v1.*;
import org.apache.spark.SparkConf;
import org.apache.spark.*;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.*;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.streaming.*;
import org.apache.spark.streaming.api.java.*;
import org.apache.spark.streaming.pubsub.PubsubUtils;
import org.apache.spark.streaming.pubsub.SparkGCPCredentials;
import org.json.JSONArray;
import org.json.JSONObject;
import scala.Tuple2;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.streaming.pubsub.SparkPubsubMessage;

import java.util.Iterator;

public class PubsubStreming {
    public static void main(String[] args) throws InterruptedException {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("NetworkWordCount");
        JavaStreamingContext jsc = new JavaStreamingContext(conf, Durations.seconds(1));

        // Set log level
        jsc.sparkContext().setLogLevel("INFO");

        JavaDStream<SparkPubsubMessage> stream = null;
        for (int i = 0; i < 5; i += 1) {
            JavaDStream<SparkPubsubMessage> pubSubReciever =
                    PubsubUtils.createStream(
                            jsc,
                            "ikame-gem-ai-research",
                            "test-1",
                            new SparkGCPCredentials.Builder().build(),
                            StorageLevel.MEMORY_AND_DISK_SER());
            if (stream == null) {
                stream = pubSubReciever;
            } else {
                stream = stream.union(pubSubReciever);
            }
        }

        writeToBQ(stream, "ikame-gem-ai-research", "Adjust_realtime", "test_2", 1);

        jsc.start();
        jsc.awaitTerminationOrTimeout(60000);

        jsc.stop();

    }
    public static void writeToBQ(
            JavaDStream<SparkPubsubMessage> pubSubStream,
            String outputProjectID,
            String pubSubBQOutputDataset,
            String PubSubBQOutputTable,
            Integer batchSize) {
        pubSubStream.foreachRDD(
                new VoidFunction<JavaRDD<SparkPubsubMessage>>() {
                    @Override
                    public void call(JavaRDD<SparkPubsubMessage> sparkPubsubMessageJavaRDD) throws Exception {
                        sparkPubsubMessageJavaRDD.foreachPartition(
                                new VoidFunction<Iterator<SparkPubsubMessage>>() {
                                    @Override
                                    public void call(Iterator<SparkPubsubMessage> sparkPubsubMessageIterator)
                                            throws Exception {
                                        System.out.println("VKLLLLLLLLLLLLL");
                                        BigQueryWriteClient bqClient = BigQueryWriteClient.create();
                                        WriteStream stream =
                                                WriteStream.newBuilder().setType(WriteStream.Type.COMMITTED).build();
                                        TableName tableName =
                                                TableName.of(outputProjectID, pubSubBQOutputDataset, PubSubBQOutputTable);
                                        CreateWriteStreamRequest createWriteStreamRequest =
                                                CreateWriteStreamRequest.newBuilder()
                                                        .setParent(tableName.toString())
                                                        .setWriteStream(stream)
                                                        .build();
                                        WriteStream writeStream = bqClient.createWriteStream(createWriteStreamRequest);
                                        System.out.println("AAAAAAAAAAAAAA");

                                        try (JsonStreamWriter writer =
                                                     JsonStreamWriter.newBuilder(
                                                                     writeStream.getName(), writeStream.getTableSchema())
                                                             .build()) {

                                            JSONArray jsonArr = new JSONArray();
                                            while (sparkPubsubMessageIterator.hasNext()) {
                                                System.out.println("asda");

                                                SparkPubsubMessage message = sparkPubsubMessageIterator.next();
                                                JSONObject record = new JSONObject(new String(message.getData()));
                                                jsonArr.put(record);
                                                if (jsonArr.length() == batchSize) {
                                                    ApiFuture<AppendRowsResponse> future = writer.append(jsonArr);
                                                    AppendRowsResponse response = future.get();
                                                    jsonArr = new JSONArray();
                                                }
                                            }
                                            if (jsonArr.length() > 0) {
                                                ApiFuture<AppendRowsResponse> future = writer.append(jsonArr);
                                                AppendRowsResponse response = future.get();
                                            }

                                            // Finalize the stream after use.
                                            FinalizeWriteStreamRequest finalizeWriteStreamRequest =
                                                    FinalizeWriteStreamRequest.newBuilder()
                                                            .setName(writeStream.getName())
                                                            .build();
                                            bqClient.finalizeWriteStream(finalizeWriteStreamRequest);
                                        }
                                    }
                                });
                    }
                });
}}
