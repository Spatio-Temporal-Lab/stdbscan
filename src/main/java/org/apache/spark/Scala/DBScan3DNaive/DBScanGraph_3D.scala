package org.apache.spark.Scala.DBScan3DNaive
import scala.annotation.tailrec

object DBScanGraph_3D{
  def apply[T](): DBScanGraph_3D[T] ={
    new DBScanGraph_3D[T](Map[T, Set[T]]())
  }
}


class DBScanGraph_3D[T] private(nodes: Map[T, Set[T]]) extends Serializable {

  /**
   * Add the given vertex `v` to  the graph
   * @param v
   * @return
   */
  def addVertex(v: T): DBScanGraph_3D[T] = {
    nodes.get(v) match {
      case Some(_) => this
      case None => new DBScanGraph_3D(nodes + (v -> Set()))
    }
  }


  /**
   * Insert an edge from `from` to `to`
   * @param from
   * @param to
   * @return
   */
  def insertEdge(from: T, to: T): DBScanGraph_3D[T] ={
    nodes.get(from) match {
      case None => new DBScanGraph_3D[T](nodes + (from -> Set(to)))
      case Some(edges) => new DBScanGraph_3D[T](nodes + (from -> (edges + to)))
    }
  }


  /**
   * build the undirected edge from one to another
   * @param one
   * @param another
   * @return
   */
  def connect(one: T, another: T): DBScanGraph_3D[T] = {
    insertEdge(one, another).insertEdge(another, one)
  }


  @tailrec
  private def getAdjacent(tovisit: Set[T], visited: Set[T], adjacent: Set[T]): Set[T] ={
    tovisit.headOption match {
      case Some(current) =>
        nodes.get(current) match {
          case Some(edges) => {
            getAdjacent(edges.diff(visited) ++ tovisit.tail, visited + current, adjacent ++ edges)
          }
          case None => getAdjacent(tovisit.tail, visited, adjacent)
        }
      case None => adjacent
    }
  }

  /**
   * Find all `vertexs` which are reachable from `from`
   * @param from
   * @return
   */
  def getConnected(from: T): Set[T] = {
    getAdjacent(Set(from), Set[T](), Set[T]()) - from
  }
}
