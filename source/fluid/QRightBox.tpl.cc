/*
 * QRightBox.tpl.cc
 *
 *  Created on: Jun 27, 2022
 *      Author: thiele
 */

#include <fluid/QRightBox.tpl.hh>

template <>
QRightBox<1>::QRightBox() : Quadrature<1>(1) {
  this->quadrature_points[0] = dealii::Point<1>(1.0);
  this->weights[0] = 1.0;
}
