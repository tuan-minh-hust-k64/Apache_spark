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
                .mode(SaveMode.Append).save("ikame-ltv-predict.ltv_prediction.Pdf_Reader_2_input_data_traning_total");
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
                .withColumn("mode__event__in_app_purchase__subscription__int_value",
                        when(col("mode__event__in_app_purchase__subscription__int_value").isNull(), 0)
                                .otherwise(col("mode__event__in_app_purchase__subscription__int_value")))
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
        mlData = mlData.drop("mode__event__ad_track__action_name__string_value",
                "mode__event__ad_track__status_result__string_value",
                "mode__event__ad_track__status_internet__string_value",
                "mode__event__ad_track__status_Ad_position__string_value",
                "mode__event__paid_ad_impression_all__ad_format__string_value",
                "mode__event_name");
        return mlData;
    }
}
