package com.apache.spark;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("apache-spark").setMaster("local[*]");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        JavaRDD<String> textFile = javaSparkContext.textFile("src/main/resources/input.txt");
        JavaPairRDD<String, Integer> counts = textFile
                .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
                .mapToPair(word -> new Tuple2<>(word.trim().toLowerCase().replaceAll("[^a-zA-Z]", ""), 1))
                .reduceByKey(Integer::sum).sortByKey();
        counts.saveAsTextFile("src/main/resources/result");
        javaSparkContext.close();
    }
}
