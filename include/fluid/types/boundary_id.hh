/**
 * @file boundary_id.hh
 *
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-05, obstacle, JR
 * @date 2019-11-11, UK
 * @date 2018-03-07, UK
 * @date 2017-11-08, UK
 * @date 2015-11-20, UK
 * @date 2015-10-26, UK
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

#ifndef __fluid_boundary_id_hh
#define __fluid_boundary_id_hh

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

namespace fluid {
namespace types {

namespace space {

// NOTE:    boundary_id: multi-physics flag similar colorisation
// WARNING: the deal.II numeric type for boundary_id should be unsigned int
//          (at least 16; mostly 32 bits sometimes 64)

enum class boundary_id : dealii::types::boundary_id {
	invalid                    = 0,         // value: 0
	
	// flow: convection field variable
	prescribed_convection_c1   = (1u << 0), // value: 1
	prescribed_convection_c2   = (1u << 1), // value: 2
	prescribed_convection_c3   = (1u << 2), // value: 4
	prescribed_no_slip         = (1u << 3), // value: 8
	prescribed_do_nothing      = (1u << 4), // value: 16
	prescribed_obstacle        = (1u << 5), // value: 32
	
// 	// flow: pressure variable
// 	prescribed_pressure        = (1u << 6), // value: 64
// 	prescribed_vol_flux        = (1u << 7), // value: 128
	
	// other
	forbidden = (1u << (8*sizeof(dealii::types::boundary_id)-1)),
	forbidden_dealii_internal_use = static_cast< dealii::types::boundary_id > (-1)
};

}

namespace time {

enum class boundary_id : unsigned int {
	forbidden = 0,
	Sigma0 = 1,
	SigmaT = 2
};

}

}}

////////////////////////////////////////////////////////////////////////////////
// Further implementations
//

namespace fluid {
namespace types {

namespace space {

using boundary_id_numeric_type = unsigned int;

inline boundary_id operator | (boundary_id id0, boundary_id id1) {
	return static_cast< boundary_id > (
		static_cast< boundary_id_numeric_type > (id0) |
		static_cast< boundary_id_numeric_type > (id1)
	);
}

inline boundary_id operator & (boundary_id id0, boundary_id id1) {
	return static_cast< boundary_id > (
		static_cast< boundary_id_numeric_type > (id0) &
		static_cast< boundary_id_numeric_type > (id1)
	);
}

inline boundary_id operator + (boundary_id id0, boundary_id id1) {
	return operator | (id0, id1);
}

inline boundary_id operator - (boundary_id /*id0*/, boundary_id /*id1*/) {
	// Do *NOT* allow the substraction of a boundary_id flag
	AssertThrow(false, dealii::ExcNotImplemented());
	return boundary_id::invalid;
}

inline bool operator == (dealii::types::boundary_id id0, boundary_id id1) {
	return
		static_cast< boundary_id_numeric_type >(id0) &
		static_cast< boundary_id_numeric_type >(id1) ? true : false;
}

inline bool operator == (boundary_id id1, dealii::types::boundary_id id0) {
	return operator == (id0, id1);
}

inline bool operator != (dealii::types::boundary_id id0, boundary_id id1) {
	return ! operator == (id0, id1);
}

inline bool operator != (boundary_id id1, dealii::types::boundary_id id0) {
	return operator != (id0, id1);
}

}}} // namespace

#endif
