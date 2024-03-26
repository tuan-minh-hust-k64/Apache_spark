package com.apache.spark;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.SaveMode;

import java.time.ZoneId;
import java.time.ZonedDateTime;

import static org.apache.spark.sql.functions.*;

public class SparkUs {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("Spark ML").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .config("spark.jars", "https://storage.cloud.google.com/spark-lib/bigquery/spark-bigquery-latest.jar,https://storage.cloud.google.com/spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.36.1.jar")
                .config("spark.jars.packages", "com.google.cloud.spark:spark-3.5-bigquery:0.36.1")
//                .config("parentProject", "ikame-ltv-predict")
                .getOrCreate();
//        sparkSession.conf().set("credentialsFile", "src/main/resources/bigquery.json");
        System.out.println(ZonedDateTime.now(ZoneId.of("UTC")));
        Dataset<Row> mlData = sparkSession.read().format("bigquery")
                .option("table","ikame-bi-segmentation.iaa_segmentation.word_office_input_data")
                .load();
        mlData = mlData.withColumn("snapshot_ts_day_of_week", regexp_replace(col("snapshot_ts_day_of_week"), "D", "").cast(DataTypes.FloatType))
                .withColumn("snapshot_ts_week_of_year", regexp_replace(col("snapshot_ts_week_of_year"), "W", "").cast(DataTypes.FloatType))
                .withColumn("snapshot_ts_month_of_year", regexp_replace(col("snapshot_ts_month_of_year"), "M", "").cast(DataTypes.FloatType));
        StringIndexer stringIndexer = new StringIndexer();
        mlData = stringIndexer.setInputCols(new String[] {
                "mode_event_ad_track_action_name_string_value",
                "mode_event_ad_track_status_result_string_value",
                "mode_event_ad_track_status_internet_string_value",
                "mode_event_ad_track_status_Ad_position_string_value",
                "mode_event_paid_ad_impression_all_ad_format_string_value",
                "mode_event_name",
                "mode_event_in_app_purchase_subscription_int_value"
        }).setOutputCols(new String[] {
                "mode_event_ad_track_action_name_string_value_index",
                "mode_event_ad_track_status_result_string_value_index",
                "mode_event_ad_track_status_internet_string_value_index",
                "mode_event_ad_track_status_Ad_position_string_value_index",
                "mode_event_paid_ad_impression_all_ad_format_string_value_index",
                "mode_event_name_index",
                "mode_event_in_app_purchase_subscription_int_value_index"
        }).setHandleInvalid("keep").fit(mlData).transform(mlData);
        mlData = mlData.withColumn("mode_event_ad_track_action_name_string_value", when(col("mode_event_ad_track_action_name_string_value").isNotNull(),
                col("mode_event_ad_track_action_name_string_value_index")).otherwise(null))
                .withColumn("mode_event_ad_track_status_result_string_value", when(col("mode_event_ad_track_status_result_string_value").isNotNull(),
                        col("mode_event_ad_track_status_result_string_value_index")).otherwise(null))
                .withColumn("mode_event_ad_track_status_internet_string_value", when(col("mode_event_ad_track_status_internet_string_value").isNotNull(),
                        col("mode_event_ad_track_status_internet_string_value_index")).otherwise(null))
                .withColumn("mode_event_ad_track_status_Ad_position_string_value", when(col("mode_event_ad_track_status_Ad_position_string_value").isNotNull(),
                        col("mode_event_ad_track_status_Ad_position_string_value_index")).otherwise(null))
                .withColumn("mode_event_paid_ad_impression_all_ad_format_string_value", when(col("mode_event_paid_ad_impression_all_ad_format_string_value").isNotNull(),
                        col("mode_event_paid_ad_impression_all_ad_format_string_value_index")).otherwise(null))
                .withColumn("mode_event_name", when(col("mode_event_name").isNotNull(),
                        col("mode_event_name_index")).otherwise(null))
                .withColumn("mode_event_in_app_purchase_subscription_int_value", when(col("mode_event_in_app_purchase_subscription_int_value").isNotNull(),
                        col("mode_event_in_app_purchase_subscription_int_value_index")).otherwise(null))
                .drop("mode_event_ad_track_action_name_string_value_index",
                        "mode_event_ad_track_status_result_string_value_index",
                        "mode_event_ad_track_status_internet_string_value_index",
                        "mode_event_ad_track_status_Ad_position_string_value_index",
                        "mode_event_paid_ad_impression_all_ad_format_string_value_index",
                        "mode_event_name_index",
                        "mode_event_in_app_purchase_subscription_int_value_index");
        mlData.write().format("bigquery").option("writeMethod", "direct")
                .mode(SaveMode.Overwrite).save("ikame-bi-segmentation.iaa_segmentation.word_office_input_data_traning");
        System.out.println(ZonedDateTime.now(ZoneId.of("UTC")));

//        mlData = mlData.filter("is_cancelled=false").drop("is_cancelled", "observation_date");
//        mlData = mlData.withColumn("next_month_views", when(col("next_month_views").$greater(0), 1).otherwise(0))
//                .withColumn("last_month_views", when(col("last_month_views").isNull(), 0).otherwise(col("last_month_views")))
//                .withColumn("all_time_views", when(col("all_time_views").isNull(), 0).otherwise(col("all_time_views")))
//                .withColumn("firstSub", when(col("firstSub").isNull(), 0).otherwise(col("firstSub")))
//                .withColumnRenamed("next_month_views", "label");
//        StringIndexer stringIndexer = new StringIndexer();
//        mlData = stringIndexer.setInputCols(new String[] {"payment_method_type", "country", "rebill_period_in_months"})
//                .setOutputCols(new String[] {"payIndex", "countryIndex", "rebillIndex"}).fit(mlData).transform(mlData);
//        OneHotEncoder oneHotEncoder = new OneHotEncoder();
//        mlData = oneHotEncoder.setInputCols(new String[] {"payIndex", "countryIndex", "rebillIndex"})
//                .setOutputCols(new String[] {"payVector", "countryVector", "rebillVector"})
//                .fit(mlData)
//                .transform(mlData);
//        mlData.show(10);
//        VectorAssembler vectorAssembler = new VectorAssembler();
//        mlData = vectorAssembler.setInputCols(new String[] {"payVector", "countryVector", "rebillVector", "firstSub", "all_time_views", "last_month_views"})
//                .setOutputCol("features").transform(mlData).select("features", "label");
//
//        Dataset<Row>[] splitRandomData = mlData.randomSplit(new double[] {0.9, 0.1});
//        Dataset<Row> trainAndHoldData = splitRandomData[0];
//        Dataset<Row> holdData = splitRandomData[1];
//
//        LogisticRegression logisticRegression = new LogisticRegression();
//        ParamMap[] paramGridBuilder = new ParamGridBuilder().addGrid(logisticRegression.regParam(), new double[] {0, 0.2, 0.4, 0.6, 0.8, 1})
//                        .addGrid(logisticRegression.elasticNetParam(), new double[] {0, 0.5, 1}).build();
//        TrainValidationSplit trainValidationSplit = new TrainValidationSplit().setEstimator(logisticRegression)
//                        .setEstimatorParamMaps(paramGridBuilder)
//                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
//                                .setTrainRatio(0.8);
//        LogisticRegressionModel logisticRegressionModel = (LogisticRegressionModel) trainValidationSplit.fit(trainAndHoldData).bestModel();
//        System.out.println("Accuracy: " + logisticRegressionModel.summary().accuracy());
//        LogisticRegressionSummary summaryHoldData = logisticRegressionModel.evaluate(holdData);
    }
}
