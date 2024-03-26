package com.apache.spark;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;


public class SparkML {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);
        SparkSession sparkSession = SparkSession.builder().config("spark.sql.warehouse.dir", "file:///c:/tmp/").appName("Spark ML")
                .master("local[*]").getOrCreate();
        Dataset<Row> rawDataset = sparkSession.read().option("header", true).option("inferSchema", true).csv("src/main/resources/vppFreeTrials.csv");
        Dataset<Row> rawDataset1 = sparkSession.read().option("header", true).option("inferSchema", true).csv("src/main/resources/vppFreeTrials1.csv");
        StringIndexer stringIndexer = new StringIndexer();
        stringIndexer.setInputCols(new String[] { "country"})
                .setOutputCols(new String[] { "country"});
        StringIndexerModel stringIndexerModel = stringIndexer.fit(rawDataset).setHandleInvalid("keep");
        rawDataset = stringIndexerModel.transform(rawDataset);
        rawDataset.show();
        rawDataset1 = stringIndexerModel.transform(rawDataset1);
        rawDataset1.show();



//        OneHotEncoder oneHotEncoder = new OneHotEncoder();
//        oneHotEncoder.setInputCols(new String[] { "conditionIndex", "gradeIndex", "zipcodeIndex", "waterfrontIndex" })
//                .setOutputCols(new String[] { "conditionVector", "gradeVector", "zipcodeVector", "waterfrontVector" });
//        rawDataset = oneHotEncoder.fit(rawDataset).transform(rawDataset);
//        rawDataset = rawDataset.withColumn("sqft_above_percentage", col("sqft_above").divide(col("sqft_living")));
//        rawDataset.show(20);
//        VectorAssembler vectorAssembler = new VectorAssembler();
//        vectorAssembler.setInputCols(new String[] { "bedrooms", "bathrooms", "sqft_living", "sqft_above_percentage", "floors", "conditionVector",
//                "gradeVector", "zipcodeVector", "waterfrontVector"}).setOutputCol("features");
//        Dataset<Row> vectorDataset = vectorAssembler.transform(rawDataset);
//        Dataset<Row> mlDataset = vectorDataset.select("price", "features").withColumnRenamed("price", "label");
//        Dataset<Row>[] mlDatasetSplits = mlDataset.randomSplit(new double[]{0.8, 0.2});
//        Dataset<Row> mlDatasetTrainAndTest = mlDatasetSplits[0];
//        Dataset<Row> mlDatasetHold = mlDatasetSplits[1];
//        LinearRegression linearRegression = new LinearRegression();
//        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
//        ParamMap[] mapParams = paramGridBuilder.addGrid(linearRegression.regParam(), new double[]{0.01, 0.1, 0.5})
//                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1})
//                .build();
//        TrainValidationSplit trainValidationSplit = new TrainValidationSplit().setEstimator(linearRegression)
//                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
//                .setEstimatorParamMaps(mapParams)
//                .setTrainRatio(0.8);
//        TrainValidationSplitModel model = trainValidationSplit.fit(mlDatasetTrainAndTest);
//        LinearRegressionModel linearRegressionModel = (LinearRegressionModel) model.bestModel();
//        System.out.println("R2: " + linearRegressionModel.summary().r2() + ", coefficients: " + linearRegressionModel.summary().rootMeanSquaredError());
    }
}
