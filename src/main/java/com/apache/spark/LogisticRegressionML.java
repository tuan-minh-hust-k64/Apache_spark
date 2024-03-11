package com.apache.spark;

import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

public class LogisticRegressionML {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder()
                .appName("SparkML")
                .master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .getOrCreate();
        Dataset<Row> mlData = sparkSession.read().option("header", true).option("inferSchema", true).csv("src/main/resources/vppChapterViews/part-r*.csv");
        mlData = mlData.filter("is_cancelled=false").drop("is_cancelled", "observation_date");
        mlData = mlData.withColumn("next_month_views", when(col("next_month_views").$greater(0), 1).otherwise(0))
                .withColumn("last_month_views", when(col("last_month_views").isNull(), 0).otherwise(col("last_month_views")))
                .withColumn("all_time_views", when(col("all_time_views").isNull(), 0).otherwise(col("all_time_views")))
                .withColumn("firstSub", when(col("firstSub").isNull(), 0).otherwise(col("firstSub")))
                .withColumnRenamed("next_month_views", "label");
        StringIndexer stringIndexer = new StringIndexer();
        mlData = stringIndexer.setInputCols(new String[] {"payment_method_type", "country", "rebill_period_in_months"})
                .setOutputCols(new String[] {"payIndex", "countryIndex", "rebillIndex"}).fit(mlData).transform(mlData);
        OneHotEncoder oneHotEncoder = new OneHotEncoder();
        mlData = oneHotEncoder.setInputCols(new String[] {"payIndex, countryIndex", "rebillIndex"})
                .setOutputCols(new String[] {"payVector, countryVector", "rebillVector"})
                .fit(mlData)
                .transform(mlData);

        VectorAssembler vectorAssembler = new VectorAssembler();
        mlData = vectorAssembler.setInputCols(new String[] {"payVector, countryVector", "rebillVector", "firstSub", "all_time_views", "last_month_view"})
                .setOutputCol("features").transform(mlData).select("features", "label");
        mlData.show(10);
    }
}
