/*
 * QRightBox.tpl.hh
 *
 *  Created on: Jun 27, 2022
 *      Author: thiele
 */

#ifndef INCLUDE_QRIGHTBOX_TPL_HH_
#define INCLUDE_QRIGHTBOX_TPL_HH_

#include<deal.II/base/quadrature.h>

template<int dim>
class QRightBox : public dealii::Quadrature<dim>
{
public:
	QRightBox();
};

template<>
QRightBox<1>::QRightBox();

#endif /* INCLUDE_QRIGHTBOX_TPL_HH_ */
