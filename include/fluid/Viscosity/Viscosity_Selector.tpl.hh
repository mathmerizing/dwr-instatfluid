/**
 * @file Viscosity_Selector.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-07, Viscosity, UK
 * @date 2018-07-30, Density, contributed by MPB from Force_Selector (UK)
 * @date 2018-07-26, dwr, UK
 * @date 2016-05-30, UK
 * @date 2016-02-11, UK
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher and contributors                   */
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

#ifndef __Viscosity_Selector_tpl_hh
#define __Viscosity_Selector_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <memory>
#include <string>

namespace fluid {
namespace viscosity {

template<int dim>
class Selector {
public:
	Selector() = default;
	virtual ~Selector() = default;
	
	virtual void create_function(
		const std::string &type,
		const std::string &options,
		std::shared_ptr< dealii::Function<dim> > &function
	) const;
};

}}

#endif
