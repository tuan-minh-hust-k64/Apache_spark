package com.apache.spark;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class SparkMLHouse {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("Spark ML").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .config("spark.jars", "https://storage.cloud.google.com/spark-lib/bigquery/spark-bigquery-latest.jar")
                .getOrCreate();
        sparkSession.conf().set("credentialsFile", "src/main/resources/bigquery.json");
        System.out.println(ZonedDateTime.now(ZoneId.of("UTC")));
        Dataset<Row> mlData = sparkSession.read().format("bigquery").option("table","ikame-ltv-predict.ltv_prediction.Pdf_Reader_2_input_data")
                .load();
        mlData = processData(mlData, "ltv_total");
        mlData.write().format("bigquery").option("writeMethod", "direct")
                .mode(SaveMode.Overwrite).save("ikame-ltv-predict.ltv_prediction.Pdf_Reader_2_input_data_traning_ltv_total");
        System.out.println(ZonedDateTime.now(ZoneId.of("UTC")));

    }

    private static Dataset<Row> processData(Dataset<Row> mlData, String filterBy) {
        String[] newColumns = Arrays.stream(mlData.columns())
                .filter(column -> column.startsWith("ltv_iap_d") || column.startsWith("ltv_ad_d"))
                .toArray(String[]::new);
        String[] labelsColumn = Arrays.stream(mlData.columns())
                .filter(column -> column.startsWith("ltv_total_d"))
                .toArray(String[]::new);
        mlData = mlData.filter(col(filterBy+"_d0").isNotNull()).drop(newColumns);
        mlData = mlData.withColumn("labels", concat(lit("["), concat_ws(", ",
                        Arrays.stream(labelsColumn).map(functions::col).toArray(Column[]::new)
                ), lit("]")).cast(DataTypes.StringType))
                .withColumn("snapshot_ts_day_of_week", regexp_replace(col("snapshot_ts_day_of_week"), "D", "").cast(DataTypes.FloatType))
                .withColumn("snapshot_ts_week_of_year", regexp_replace(col("snapshot_ts_week_of_year"), "W", "").cast(DataTypes.FloatType))
                .withColumn("snapshot_ts_month_of_year", regexp_replace(col("snapshot_ts_month_of_year"), "M", "").cast(DataTypes.FloatType))
                .withColumn(filterBy + "_d0_feature", when(col("snapshot_ts_day_of_week").$greater$eq(1), col(filterBy + "_d0")).otherwise(null))
                .withColumn(filterBy + "_d1_feature", when(col("snapshot_ts_day_of_week").$greater$eq(2), col(filterBy + "_d1")).otherwise(null))
                .withColumn(filterBy + "_d2_feature", when(col("snapshot_ts_day_of_week").$greater$eq(3), col(filterBy + "_d2")).otherwise(null))
                .withColumn(filterBy + "_d3_feature", when(col("snapshot_ts_day_of_week").$greater$eq(4), col(filterBy + "_d3")).otherwise(null))
                .withColumn(filterBy + "_d4_feature", when(col("snapshot_ts_day_of_week").$greater$eq(5), col(filterBy + "_d4")).otherwise(null))
                .withColumn(filterBy + "_d5_feature", when(col("snapshot_ts_day_of_week").$greater$eq(6), col(filterBy + "_d5")).otherwise(null))
                .withColumn(filterBy + "_d6_feature", when(col("snapshot_ts_day_of_week").$greater$eq(7), col(filterBy + "_d6")).otherwise(null))
        ;
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
        mlData = mlData.drop("mode_event_ad_track_action_name_string_value",
                "mode_event_ad_track_status_result_string_value",
                "mode_event_ad_track_status_internet_string_value",
                "mode_event_ad_track_status_Ad_position_string_value",
                "mode_event_paid_ad_impression_all_ad_format_string_value",
                "mode_event_name",
                "mode_event_in_app_purchase_subscription_int_value");
        return mlData;
    }
}
