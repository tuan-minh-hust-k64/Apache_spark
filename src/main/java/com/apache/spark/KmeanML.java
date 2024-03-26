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
//                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .config("spark.jars", "https://storage.cloud.google.com/spark-lib/bigquery/spark-bigquery-latest.jar")
                .config("parentProject", "ikame-pltv-project")
                .getOrCreate();
        sparkSession.conf().set("credentialsFile", "/home/minhvt/bigquery.json");
        System.out.println(ZonedDateTime.now(ZoneId.of("UTC")));
        Dataset<Row> dataset = sparkSession.read().format("bigquery")
                .option("parentProject", "ikame-pltv-project")
                .option("table","ikame-pltv-project.Dino_Moto_Race_custom_model_pltv_activation_automation.pre_new_cohort_SCORE")
                .load();
        dataset.show(1);
        System.out.println(ZonedDateTime.now(ZoneId.of("UTC")));
//        Dataset<Row> df_exploded = dataset.selectExpr("*", "posexplode(event_params) as (p, event_params_key, x)")
//                .withColumn("string_value", col("x").getItem("string_value"))
//                .drop("p", "x", "event_params");
//        df_exploded.createOrReplaceTempView("AnalyticTable");
//        Dataset<Row> transformedDF = sparkSession.sql("SELECT CONCAT(user_pseudo_id, '/', event_date) AS session_id, " +
//                "user_pseudo_id AS user_id, " +
//                "TIMESTAMP_MICROS(event_timestamp) AS ts, " +
//                "CONCAT(event_name, '_string_', event_params_key) AS name, " +
//                "string_value AS value " +
//                "FROM AnalyticTable " +
//                "WHERE string_value IS NOT NULL");
//
//        // Hiển thị kết quả
//        transformedDF.write().format("bigquery").option("writeMethod", "direct")
//                .mode(SaveMode.Overwrite)
//                .save("ikame-pltv-project.Dino_Moto_Race_custom_model_pltv_activation_automation.categorical_facts_data_preparation_spark");

//        dataset.write().format("bigquery").option("writeMethod", "direct")
//                .mode(SaveMode.Append).save("ikame-ltv-predict.ltv_prediction.ltv_data_training_test");
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
