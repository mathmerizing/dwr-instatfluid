/**
 * @file   ParameterHandler.hh
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * @author Julian Roth (JR)
 * 
 * @Date 2022-04-26, high/low order problem, JR
 * @Date 2022-01-17, Added Newton parameters, JPT
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-07, stokes, UK
 * @date 2018-07-25, new parameters dwr, UK
 * @date 2018-03-06, included from ewave, UK
 * @date 2017-02-06, UK
 */

/*  Copyright (C) 2012-2022 by Uwe Koecher and contributors                   */
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

#ifndef __ParameterHandler_hh
#define __ParameterHandler_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes

namespace fluid {

class ParameterHandler : public dealii::ParameterHandler {
public:
	ParameterHandler();
	virtual ~ParameterHandler() = default;
};


} // namespace

#endif
