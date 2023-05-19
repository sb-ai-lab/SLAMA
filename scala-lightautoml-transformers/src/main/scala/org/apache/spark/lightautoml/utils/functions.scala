package org.apache.spark.lightautoml.utils

import breeze.linalg
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

object functions {
  val vectorAveragingUdf: UserDefinedFunction = udf { (vecs: Seq[DenseVector], vec_size: Int) =>
    val not_null_vecs = vecs.count(_ != null)
    if (not_null_vecs == 0){
      //      null
      new DenseVector(linalg.DenseVector.fill(vec_size)(0.0).toArray)
    }
    else {
      val result = vecs.map { vec: Any =>
        vec match {
          case v if v == null => linalg.DenseVector.fill(vec_size)(0.0)
          case v: Vector => v.asBreeze
          case v => throw new IllegalArgumentException(
            "function vector_to_array requires a non-null input argument and input type must be " +
                    "`org.apache.spark.ml.linalg.Vector` or `org.apache.spark.mllib.linalg.Vector`, " +
                    s"but got ${if (v == null) "null" else v.getClass.getName}.")
        }
      }.reduce(_ + _)
      val v = result.mapActiveValues(_ / not_null_vecs)
      new DenseVector(v.toArray)
    }
  }

  val scalarAveragingUdf: UserDefinedFunction = udf { (vecs: Seq[Option[Double]]) =>
    vecs.aggregate((0.0, 0))(
      (agg, optVal) => optVal match {
        case Some(v) => (agg._1 + v, agg._2 + 1)
        case _ => agg
      },
      (x, y) => (x._1 + y._1, x._2 + y._2)
    ) match {
      case (_, 0) => 0
      case (sm, cnt) => sm / cnt
    }


//    val not_null_vecs = vecs.count(_.nonEmpty)
//    if (not_null_vecs == 0){
////      null
//      0.0
//    }
//    else {
//      val result = vecs.map {
//        case None => 0.0
//        case Some(v) => v
//      }.sum
//      result / not_null_vecs
//    }
  }


  def vector_averaging(vecs_col: Column, dim_size: Column): Column = {
    vectorAveragingUdf(vecs_col, dim_size)
  }

  def scalar_averaging(scalars_col: Column): Column = {
    scalarAveragingUdf(scalars_col)
  }
}
