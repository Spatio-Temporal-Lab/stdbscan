package org.apache.spark.Scala.utils.partition

import scala.collection.mutable

case class Graph(vertices: mutable.SortedSet[Int], edges: Map[(Int, Int), Double])
