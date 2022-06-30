/**
 * @file Grid_Schaefer_Turek_2D.tpl.hh
 * @author Julian Roth (JR)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-09-27, Schaefer/Turek 2D, JR
 * 
 * @date 2019-11-11, UK
 * @date 2018-07-26, UK
 * @date 2018-03-06, UK
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

#ifndef __Grid_Cavity_2D_tpl_hh
#define __Grid_Cavity_2D_tpl_hh

// PROJECT includes
#include <fluid/grid/Grid.tpl.hh>
#include <fluid/parameters/ParameterSet.hh>

namespace fluid {
namespace grid {

template<int dim>
class Grid_Cavity_2D :
	public fluid::Grid<dim> {
public:
	Grid_Cavity_2D(
		std::shared_ptr< fluid::ParameterSet > parameter_set
	) :
		fluid::Grid<dim> (parameter_set
//				,true
				)
	{};
	
	virtual ~Grid_Cavity_2D() = default;
	
	virtual void set_manifolds();
	virtual void set_boundary_indicators();
};

}} // namespace

#endif
