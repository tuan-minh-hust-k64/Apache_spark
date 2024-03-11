package com.apache.spark;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

import java.util.List;

import static org.apache.spark.sql.functions.*;

public class SparkMLHouse {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder()
                .appName("SparkML")
                .master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .getOrCreate();
        Dataset<Row> mlData = sparkSession.read().option("header", true).option("inferSchema", true).csv("src/main/resources/ltv/raw-data-*");
        mlData = mlData.filter(col("ltv_ad_d0").isNotNull()).drop(
                "ltv_ad_d31",
                "ltv_ad_d32",
                "ltv_ad_d33",
                "ltv_ad_d34",
                "ltv_ad_d35",
                "ltv_ad_d36",
                "ltv_ad_d37",
                "ltv_ad_d38",
                "ltv_ad_d39",
                "ltv_ad_d40",
                "ltv_ad_d41",
                "ltv_ad_d42",
                "ltv_ad_d43",
                "ltv_ad_d44",
                "ltv_ad_d45",
                "ltv_ad_d46",
                "ltv_ad_d47",
                "ltv_ad_d48",
                "ltv_ad_d49",
                "ltv_ad_d50",
                "ltv_ad_d51",
                "ltv_ad_d52",
                "ltv_ad_d53",
                "ltv_ad_d54",
                "ltv_ad_d55",
                "ltv_ad_d56",
                "ltv_ad_d57",
                "ltv_ad_d58",
                "ltv_ad_d59",
                "ltv_ad_d60",
                "ltv_iap_d0",
                "ltv_iap_d1",
                "ltv_iap_d2",
                "ltv_iap_d3",
                "ltv_iap_d4",
                "ltv_iap_d5",
                "ltv_iap_d6",
                "ltv_iap_d7",
                "ltv_iap_d8",
                "ltv_iap_d9",
                "ltv_iap_d10",
                "ltv_iap_d11",
                "ltv_iap_d12",
                "ltv_iap_d13",
                "ltv_iap_d14",
                "ltv_iap_d15",
                "ltv_iap_d16",
                "ltv_iap_d17",
                "ltv_iap_d18",
                "ltv_iap_d19",
                "ltv_iap_d20",
                "ltv_iap_d21",
                "ltv_iap_d22",
                "ltv_iap_d23",
                "ltv_iap_d24",
                "ltv_iap_d25",
                "ltv_iap_d26",
                "ltv_iap_d27",
                "ltv_iap_d28",
                "ltv_iap_d29",
                "ltv_iap_d30",
                "ltv_iap_d31",
                "ltv_iap_d32",
                "ltv_iap_d33",
                "ltv_iap_d34",
                "ltv_iap_d35",
                "ltv_iap_d36",
                "ltv_iap_d37",
                "ltv_iap_d38",
                "ltv_iap_d39",
                "ltv_iap_d40",
                "ltv_iap_d41",
                "ltv_iap_d42",
                "ltv_iap_d43",
                "ltv_iap_d44",
                "ltv_iap_d45",
                "ltv_iap_d46",
                "ltv_iap_d47",
                "ltv_iap_d48",
                "ltv_iap_d49",
                "ltv_iap_d50",
                "ltv_iap_d51",
                "ltv_iap_d52",
                "ltv_iap_d53",
                "ltv_iap_d54",
                "ltv_iap_d55",
                "ltv_iap_d56",
                "ltv_iap_d57",
                "ltv_iap_d58",
                "ltv_iap_d59",
                "ltv_iap_d60",
                "ltv_total_d0",
                "ltv_total_d1",
                "ltv_total_d2",
                "ltv_total_d3",
                "ltv_total_d4",
                "ltv_total_d5",
                "ltv_total_d6",
                "ltv_total_d7",
                "ltv_total_d8",
                "ltv_total_d9",
                "ltv_total_d10",
                "ltv_total_d11",
                "ltv_total_d12",
                "ltv_total_d13",
                "ltv_total_d14",
                "ltv_total_d15",
                "ltv_total_d16",
                "ltv_total_d17",
                "ltv_total_d18",
                "ltv_total_d19",
                "ltv_total_d20",
                "ltv_total_d21",
                "ltv_total_d22",
                "ltv_total_d23",
                "ltv_total_d24",
                "ltv_total_d25",
                "ltv_total_d26",
                "ltv_total_d27",
                "ltv_total_d28",
                "ltv_total_d29",
                "ltv_total_d30",
                "ltv_total_d31",
                "ltv_total_d32",
                "ltv_total_d33",
                "ltv_total_d34",
                "ltv_total_d35",
                "ltv_total_d36",
                "ltv_total_d37",
                "ltv_total_d38",
                "ltv_total_d39",
                "ltv_total_d40",
                "ltv_total_d41",
                "ltv_total_d42",
                "ltv_total_d43",
                "ltv_total_d44",
                "ltv_total_d45",
                "ltv_total_d46",
                "ltv_total_d47",
                "ltv_total_d48",
                "ltv_total_d49",
                "ltv_total_d50",
                "ltv_total_d51",
                "ltv_total_d52",
                "ltv_total_d53",
                "ltv_total_d54",
                "ltv_total_d55",
                "ltv_total_d56",
                "ltv_total_d57",
                "ltv_total_d58",
                "ltv_total_d59",
                "ltv_total_d60"
        );
        mlData = mlData.withColumn("labels", concat(
                col("ltv_ad_d0"),
                col("ltv_ad_d1"),
                col("ltv_ad_d2"),
                col("ltv_ad_d3"),
                col("ltv_ad_d4"),
                col("ltv_ad_d5"),
                col("ltv_ad_d6"),
                col("ltv_ad_d7"),
                col("ltv_ad_d8"),
                col("ltv_ad_d9"),
                col("ltv_ad_d10"),
                col("ltv_ad_d11"),
                col("ltv_ad_d12"),
                col("ltv_ad_d13"),
                col("ltv_ad_d14"),
                col("ltv_ad_d15"),
                col("ltv_ad_d16"),
                col("ltv_ad_d17"),
                col("ltv_ad_d18"),
                col("ltv_ad_d19"),
                col("ltv_ad_d20"),
                col("ltv_ad_d21"),
                col("ltv_ad_d22"),
                col("ltv_ad_d23"),
                col("ltv_ad_d24"),
                col("ltv_ad_d25"),
                col("ltv_ad_d26"),
                col("ltv_ad_d27"),
                col("ltv_ad_d28"),
                col("ltv_ad_d29"),
                col("ltv_ad_d30")
        ))
                .withColumn("snapshot_ts_day_of_week", regexp_replace(col("snapshot_ts_day_of_week"), "D", "").cast(DataTypes.FloatType))
                .withColumn("snapshot_ts_week_of_year", regexp_replace(col("snapshot_ts_week_of_year"), "W", "").cast(DataTypes.FloatType))
                .withColumn("snapshot_ts_month_of_year", regexp_replace(col("snapshot_ts_month_of_year"), "M", "").cast(DataTypes.FloatType))
                .withColumn("mode__event__in_app_purchase__subscription__int_value",
                        when(col("mode__event__in_app_purchase__subscription__int_value").isNull(), 0)
                                .otherwise(col("mode__event__in_app_purchase__subscription__int_value")))
                .withColumn("ltv_ad_d0", when(col("snapshot_ts_day_of_week").$greater$eq(1), col("ltv_ad_d0")).otherwise(null))
                .withColumn("ltv_ad_d1", when(col("snapshot_ts_day_of_week").$greater$eq(2), col("ltv_ad_d1")).otherwise(null))
                .withColumn("ltv_ad_d2", when(col("snapshot_ts_day_of_week").$greater$eq(3), col("ltv_ad_d2")).otherwise(null))
                .withColumn("ltv_ad_d3", when(col("snapshot_ts_day_of_week").$greater$eq(4), col("ltv_ad_d3")).otherwise(null))
                .withColumn("ltv_ad_d4", when(col("snapshot_ts_day_of_week").$greater$eq(5), col("ltv_ad_d4")).otherwise(null))
                .withColumn("ltv_ad_d5", when(col("snapshot_ts_day_of_week").$greater$eq(6), col("ltv_ad_d5")).otherwise(null))
                .withColumn("ltv_ad_d6", when(col("snapshot_ts_day_of_week").$greater$eq(7), col("ltv_ad_d6")).otherwise(null))
        ;
        StringIndexer stringIndexer = new StringIndexer();
        mlData = stringIndexer.setInputCols(new String[] {
                "mode__event__ad_track__action_name__string_value",
                "mode__event__ad_track__status_result__string_value",
                "mode__event__ad_track__status_internet__string_value",
                "mode__event__ad_track__status_Ad_position__string_value",
                "mode__event__paid_ad_impression_all__ad_format__string_value",
                "mode__event_name"
        }).setOutputCols(new String[] {
                "mode__event__ad_track__action_name__string_value_index",
                "mode__event__ad_track__status_result__string_value_index",
                "mode__event__ad_track__status_internet__string_value_index",
                "mode__event__ad_track__status_Ad_position__string_value_index",
                "mode__event__paid_ad_impression_all__ad_format__string_value_index",
                "mode__event_name_index"
        }).setHandleInvalid("keep").fit(mlData).transform(mlData);
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
