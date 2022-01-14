/**
 * @file Convection_Parabolic_Inflow_2.tpl.hh
 * @author Uwe Koecher (UK)
 * 
 * @date 2019-11-12, UK
 * @date 2016-05-26, UK
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

#ifndef __Convection_Parabolic_Inflow_2_tpl_hh
#define __Convection_Parabolic_Inflow_2_tpl_hh

// DEAL.II includes
#include <deal.II/base/tensor_function.h>

// C++ includes
#include <array>

namespace convection {
namespace dirichlet {

template<int dim>
class Parabolic_Inflow_2 : public dealii::TensorFunction<1,dim> {
public:
	Parabolic_Inflow_2(
		const double &y_min,
		const double &y_max,
		const double &scaling
	);
	
	virtual ~Parabolic_Inflow_2() = default;
	
	virtual dealii::Tensor<1,dim> value(
		const dealii::Point<dim> &x
	) const;
	
private:
	const double y_min;
	const double y_max;
	std::array<double,3> a_y; // coefficients for y-profile quad. polynomial
};

}}

#endif
