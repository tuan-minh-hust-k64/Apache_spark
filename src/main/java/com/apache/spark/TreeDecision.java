package com.apache.spark;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

import java.util.Arrays;
import java.util.List;

public class TreeDecision {
    public static UDF1<String,String> countryGrouping = new UDF1<String,String>() {

        @Override
        public String call(String country) throws Exception {
            List<String> topCountries =  Arrays.asList("GB","US","IN","UNKNOWN");
            List<String> europeanCountries =  Arrays.asList("BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","CH","IS","NO","LI","EU");

            if (topCountries.contains(country)) return country;
            if (europeanCountries .contains(country)) return "EUROPE";
            else return "OTHER";
        }

    };

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("Spark ML").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
                .getOrCreate();
        Dataset<Row> rawData = sparkSession.read().option("header", true).option("inferSchema", true)
                .csv("src/main/resources/vppFreeTrials.csv");
        sparkSession.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);
        rawData = rawData.withColumn("country", call_udf("countryGrouping", col("country")))
                .withColumn("payments_made", when(col("payments_made").geq(1), lit(1)).otherwise(lit(0)));
        StringIndexer stringIndexer = new StringIndexer();
        rawData = stringIndexer.setInputCols(new String[] {"country"}).setOutputCols(new String[] {"countryIndex"}).fit(rawData).transform(rawData);
        IndexToString indexToString = new IndexToString();
        indexToString.setInputCol("countryIndex").setOutputCol("countryText").transform(rawData).select("countryIndex", "countryText").distinct().show();
        VectorAssembler vectorAssembler = new VectorAssembler();
        rawData = vectorAssembler.setInputCols(new String[] {"rebill_period", "chapter_access_count", "seconds_watched", "countryIndex"})
                        .setOutputCol("features").transform(rawData).withColumn("label", col("payments_made"))
                        .select("label", "features");
        Dataset<Row>[] trainAndHoldData = rawData.randomSplit(new double[] { 0.9, 0.1 });
        Dataset<Row> trainAndTestData = trainAndHoldData[0];
        Dataset<Row> holdData = trainAndHoldData[1];
        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();
        decisionTreeClassifier.setMaxDepth(3);
        DecisionTreeClassificationModel decisionTreeClassificationModel = decisionTreeClassifier.fit(trainAndTestData);
        Dataset<Row> predict = decisionTreeClassificationModel.transform(holdData);
        MulticlassClassificationEvaluator multiclassClassificationEvaluator = new MulticlassClassificationEvaluator();
        multiclassClassificationEvaluator.setMetricName("accuracy");
        System.out.println(multiclassClassificationEvaluator.evaluate(predict));
    }
}
