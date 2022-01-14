/**
 * @file Convection_Parabolic_Inflow_2.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-12, UK
 * @date 2016-05-26, UK
 * @date 2016-05-09, UK
 * @date 2016-01-19, UK
 * @date 2015-11-26, UK
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher                                    */
/*                                                                            */
/*  This file is part of DTM++.                                               */
/*                                                                            */
/*  DTM++ is free software: you can redistribute it and/or modify             */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++ is distributed in the hope that it will be useful,                  */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++.   If not, see <http://www.gnu.org/licenses/>.           */

#include <fluid/ConvectionDirichletBoundary/Convection_Parabolic_Inflow_2.tpl.hh>

// DEAL.II includes
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

namespace convection {
namespace dirichlet {

template<int dim>
Parabolic_Inflow_2<dim>::
Parabolic_Inflow_2(
	const double &y_min,
	const double &y_max,
	const double &scaling
) :
	y_min(y_min),
	y_max(y_max)
{
	Assert(
		(y_min < y_max),
		dealii::ExcMessage("y_min must be smaller than y_max")
	);
	
	dealii::FullMatrix<double> A(3);
	dealii::Vector<double> b; b.reinit(3);
	dealii::Vector<double> x; x.reinit(3);
	
	A(0,0) = 1.; A(0,1) = y_min; A(0,2) = y_min*y_min;
	A(1,0) = 1.; A(1,1) = y_max; A(1,2) = y_max*y_max;
	
	A(2,0) = 1.;
	A(2,1) = y_min+(y_max-y_min)/2.;
	A(2,2) = (y_min+(y_max-y_min)/2.)*(y_min+(y_max-y_min)/2.);
#ifdef DEBUG
	A.print_formatted(std::cout);
#endif
	
	b = 0.;
	b(2) = 1.;
#ifdef DEBUG
	b.print(std::cout);
#endif
	
	x = 0.;
	A.gauss_jordan();
	A.vmult(x, b);
#ifdef DEBUG
	x.print(std::cout);
#endif
	
	for (unsigned int i{0}; i < 3; ++i)
		a_y[i] = x(i)*scaling;
}

template<int dim>
dealii::Tensor<1,dim>
Parabolic_Inflow_2<dim>::
value(
	const dealii::Point<dim> &x
) const {
	Assert((dim==2), dealii::ExcNotImplemented());
	
	dealii::Tensor<1,dim> y;
	
	if (dim==2) {
		if ((x[1] >= y_min) && (x[1] <= y_max)) {
			y[0] = a_y[0] + (a_y[1] + a_y[2]*x[1])*x[1];
			y[1] = 0.;
		}
	}
	
	return y;
}

}}

#include "Convection_Parabolic_Inflow_2.inst.in"
