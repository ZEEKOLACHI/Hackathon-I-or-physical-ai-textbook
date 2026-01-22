/**
 * Custom navbar item component for UserMenu
 */

import React from 'react';
import { UserMenu } from '../../components/Auth';

// Custom component for the auth item in navbar
export function AuthNavbarItem(): JSX.Element {
  return <UserMenu />;
}

export default AuthNavbarItem;
