import { Link } from "react-router-dom";
import {
  Disclosure,
  Menu,
  MenuButton,
  MenuItem,
  MenuItems,
  Button,
} from "@headlessui/react";
import { useAuth } from "../../hooks/useAuth";
import { navigation } from "../../config/navigation";
import { FaUser } from "react-icons/fa";

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <Disclosure as="nav" className="bg-gray-800">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center">
            <div className="shrink-0">
              <img
                alt="Your Company"
                src="https://tailwindui.com/plus/img/logos/mark.svg?color=indigo&shade=500"
                className="size-8"
              />
            </div>
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    to={item.to}
                    className={classNames(
                      item.current
                        ? "bg-gray-900 text-white"
                        : "text-gray-300 hover:bg-gray-700 hover:text-white",
                      "rounded-md px-3 py-2 text-sm font-medium"
                    )}
                  >
                    {item.name}
                  </Link>
                ))}
              </div>
            </div>
          </div>
          <div className="hidden md:block">
            <div className="ml-4 flex items-center md:ml-6">
              {/* Profile dropdown */}
              {user ? (
                <Menu as="div" className="relative ml-3">
                  <div>
                    <MenuButton className="relative flex max-w-xs items-center rounded-full bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-800">
                      <span className="absolute -inset-1.5" />
                      <span className="sr-only">Open user menu</span>
                      {/* <img
                        alt=""
                        src={user.imageUrl}
                        className="size-8 rounded-full"
                      /> */}
                      <FaUser className="size-8 rounded-full" />
                    </MenuButton>
                  </div>
                  <MenuItems
                    transition
                    className="absolute right-0 z-10 mt-2 w-48 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black/5 transition focus:outline-none data-[closed]:scale-95 data-[closed]:transform data-[closed]:opacity-0 data-[enter]:duration-100 data-[leave]:duration-75 data-[enter]:ease-out data-[leave]:ease-in"
                  >
                    {/* {userNavigation.map((item) => (
                      <MenuItem key={item.name}>
                        <a
                          href={item.href}
                          className="block px-4 py-2 text-sm text-gray-700 data-[focus]:bg-gray-100 data-[focus]:outline-none"
                        >
                          {item.name}
                        </a>
                      </MenuItem>
                    ))} */}
                    <MenuItem>
                      <Link
                        to="/profile"
                        className="block px-4 py-2 text-sm text-gray-700 data-[focus]:bg-gray-100 data-[focus]:outline-none"
                      >
                        Your Profile
                      </Link>
                    </MenuItem>
                    <MenuItem>
                      <Link
                        to="/settings"
                        className="block px-4 py-2 text-sm text-gray-700 data-[focus]:bg-gray-100 data-[focus]:outline-none"
                      >
                        Settings
                      </Link>
                    </MenuItem>
                    <MenuItem>
                      <Button
                        onClick={logout}
                        className="block px-4 py-2 text-sm text-gray-700 bg-white data-[focus]:bg-gray-100 data-[focus]:outline-none min-w-full text-left border-0 rounded-none"
                      >
                        Logout
                      </Button>
                    </MenuItem>
                  </MenuItems>
                </Menu>
              ) : (
                <Link
                  to="/login"
                  className="bg-blue-500 text-white px-4 py-2 rounded-md"
                >
                  Login
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    </Disclosure>
  );
}
