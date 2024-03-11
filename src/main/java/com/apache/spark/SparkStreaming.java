package com.apache.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

public class SparkStreaming {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Spark Streaming").setMaster("local[*]");
    }
}
