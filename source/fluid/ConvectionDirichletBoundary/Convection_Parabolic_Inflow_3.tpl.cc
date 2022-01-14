/**
 * @file Convection_Parabolic_Inflow_3.tpl.cc
 * @author Uwe Köcher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-09-23, JR
 * @date 2019-11-12, UK
 * @date 2016-05-26, UK
 * @date 2016-05-09, UK
 * @date 2016-01-19, UK
 * @date 2015-11-26, UK
 */

/*  Copyright (C) 2012-2021 by Uwe Koecher and contributors                   */
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

#include <fluid/ConvectionDirichletBoundary/Convection_Parabolic_Inflow_3.tpl.hh>

namespace convection {
namespace dirichlet {

template<int dim>
dealii::Tensor<1,dim>
Parabolic_Inflow_3<dim>::
value(
	const dealii::Point<dim> &x
) const {
	Assert(((dim==2)/*||(dim==3)*/), dealii::ExcNotImplemented());
	
	dealii::Tensor<1,dim> y;
	
	// NOTE: maximal velocity for
	// --> NSE 2D-1: 0.3 m/s
	// --> NSE 2D-2: 1.5 m/s
	if (dim==2) {
		y[0] = -1. * max_velocity * (4.0 / 0.1681) *
                (std::pow(x(1), 2) - 0.41 * std::pow(x(1), 1));
		y[1] = 0.;
	}
	else {
		// TODO
	}

	return y;
}

}}

#include "Convection_Parabolic_Inflow_3.inst.in"
