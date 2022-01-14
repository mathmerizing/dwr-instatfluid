/**
 * @file Grid_Selector.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-11, UK
 * @date 2018-07-26, included from biot for dwr, UK
 * @date 2016-02-12, UK
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

#ifndef __fluid_Grid_Selector_tpl_hh
#define __fluid_Grid_Selector_tpl_hh

#include <fluid/grid/Grid.tpl.hh>
#include <fluid/parameters/ParameterSet.hh>

// C++ includes
#include <memory>
#include <string>

namespace fluid {
namespace grid {

template<int dim>
class Selector {
public:
	Selector() = default;
	virtual ~Selector() = default;
	
	virtual void create_grid(
		std::shared_ptr< fluid::ParameterSet > parameter_set,
		std::shared_ptr< fluid::Grid<dim> > &grid
	) const;
};

}}

#endif
