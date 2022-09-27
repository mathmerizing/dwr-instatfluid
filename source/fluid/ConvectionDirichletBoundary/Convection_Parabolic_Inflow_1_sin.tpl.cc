/**
 * @file Convection_Parabolic_Inflow_1_sin.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 *
 * @Date 2022-09-27, Time-dependent inflow, JR
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-12, UK
 * @date 2016-05-26, UK
 * @date 2016-05-09, UK
 * @date 2016-01-19, UK
 * @date 2015-11-26, UK
 */

/*  Copyright (C) 2012-2022 by Uwe Koecher                                    */
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

#include <fluid/ConvectionDirichletBoundary/Convection_Parabolic_Inflow_1_sin.tpl.hh>

namespace convection {
namespace dirichlet {

template<int dim>
dealii::Tensor<1,dim>
Parabolic_Inflow_1_sin<dim>::
value(
	const dealii::Point<dim> &x
) const {
	Assert(((dim==2)||(dim==3)), dealii::ExcNotImplemented());
	
	const double t{this->get_time()};

	dealii::Tensor<1,dim> y;
	
	if (dim==2) {
		if ((x[1] >= .5) && (x[1] <= 1.)) {
			y[0] = scaling * ( -8.+(24-16*x[1])*x[1] );
			y[1] = 0.;
		}
	}
	else {
		if ((x[0] >= .5) && (x[0] <= 1.) && (x[2] >= .5) && (x[2] <= 1.)) {
			y[0] = 0.;
			y[1] = scaling * (-8.+(24-16*x[0])*x[0]) * (-8.+(24-16*x[2])*x[2]);
			y[2] = 0.;
		}
	}

	y *= std::sin(M_PI * t / 8.);
	return y;
}

}}

#include "Convection_Parabolic_Inflow_1_sin.inst.in"
