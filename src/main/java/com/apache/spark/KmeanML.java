package com.apache.spark;

import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class KmeanML {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("Spark ML")
                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .config("spark.jars", "https://storage.cloud.google.com/spark-lib/bigquery/spark-bigquery-latest.jar,https://storage.cloud.google.com/spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.36.1.jar")
                .config("spark.jars.packages", "com.google.cloud.spark:spark-3.5-bigquery:0.36.1")
//                .config("parentProject", "ikame-ltv-predict")
                .getOrCreate();
        sparkSession.conf().set("credentialsFile", "/opt/bitnami/spark/test.json");
        Dataset<Row> dataset = sparkSession.read().format("bigquery")
                .option("table","bigquery-public-data.usa_names.usa_1910_2013")
                .load();
        dataset.show(1);
//        dataset.write().format("bigquery").option("writeMethod", "direct")
//                .mode(SaveMode.Append).save("ikame-ltv-predict.ltv_prediction.cast_glitter_input_data");
//        Dataset<Row> rawData = sparkSession.read().option("header", true).option("inferSchema", true)
//                .csv("src/main/resources/VPPcourseViews.csv");
//        rawData = rawData.withColumn("rate", col("proportionWatched").multiply(100)).drop("proportionWatched");
//        ALS als = new ALS();
//        als.setMaxIter(10);
//        als.setRegParam(0.1);
//        als.setUserCol("userId").setItemCol("courseId").setRatingCol("rate");
//        ALSModel alsModel = als.fit(rawData);
//        Dataset<Row> userRecs = alsModel.recommendForAllUsers(5);
//        List<Row> userRecsList = userRecs.takeAsList(5);
//        for (Row row : userRecsList) {
//            System.out.println("User " + row.getAs(0) + " , we recommend you learn: " + row.getAs(1).toString());
//
//        }
    }
}
